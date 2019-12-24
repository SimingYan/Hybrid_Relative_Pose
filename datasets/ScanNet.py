import torch.utils.data as data
import numpy as np
import torch
import cv2
import config
import os
import glob
import sys
sys.path.append("../")
from utils.img import Crop
from util import Rnd, Flip, rot2Quaternion,angular_distance_np
import util
from sklearn.neighbors import KDTree


class ScanNet(data.Dataset):
  def __init__(self, split, nViews, imgSize=224, AuthenticdepthMap=False, crop=False, cache=True,\
        hmap=False,CorresCoords=False,meta=False,rotate=False,rgbd=False,birdview=False,pointcloud=False,num_points=None,\
        denseCorres=False,segm=False,reproj=False,singleView=True,normal=False,list_=None,\
        representation='skybox',dynamicWeighting=False,snumclass=0,fullsize_rgbdn=False,entrySplit=None, corner=False,plane=False,plane_r=False,plane_m=False, scannet_new_name=0):
    self.crop = crop
    self.pointcloud = pointcloud
    self.birdview = birdview
    self.num_points = num_points
    self.rgbd = rgbd
    self.rotate = rotate
    self.meta = meta
    self.AuthenticdepthMap = AuthenticdepthMap
    self.hmap = hmap
    self.CorresCoords = CorresCoords
    self.split = split
    self.nViews = nViews
    self.imgSize = imgSize
    self.normal = normal
    self.reproj = reproj
    self.singleView = singleView
    self.snumclass = snumclass
    self.list=list_
    self.denseCorres=denseCorres
    self.segm = segm
    self.fullsize_rgbdn = fullsize_rgbdn
    self.dynamicWeighting = dynamicWeighting
    self.plane_r = plane_r
    self.plane_m = plane_m
    self.scannet_new_name = scannet_new_name
    
    self.split = 'train'
    if self.dynamicWeighting:
      assert(self.segm == True)
    self.representation = representation
    self.entrySplit=entrySplit
    if self.representation == 'skybox':
      self.OutputSize = (640,160)
    
    self.Inputwidth = config.pano_width
    self.Inputheight = config.pano_height
    self.nPanoView = 4
    self.intrinsic = np.array([[571.623718/640,0,319.500000/640],[0,571.623718/480,239.500000/480],[0,0,1]])
    self.intrinsicUnNorm = np.array([[571.623718,0,319.500000],[0,571.623718,239.500000],[0,0,1]])

    if 'scannet_test_scenes' in self.list:
        self.dataList = np.load(self.list,allow_pickle=True)
        if os.path.exists('/scratch'):
            for i in range(len(self.dataList)):
                self.dataList[i]['base'] = '/scratch/cluster/siming/yzp12' + self.dataList[i]['base'].split('SkyBox')[-1]

    else:
        self.dataList = np.load(self.list, allow_pickle=True).item()[self.split]
    
    if self.entrySplit is not None:
        self.dataList = [self.dataList[kk] for kk in range(self.entrySplit*100,(self.entrySplit+1)*100)]
    print(len(self.dataList))
    
    if self.plane_m:

        newList = []
        for i in range(len(self.dataList)):
            tp = self.dataList[i]['base']
            scene_id = tp.split('/')[-1]
            if 'scene0009_01' not in scene_id:
                continue
            plane_file = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/ScanNet_manual_plane/%s/' % self.split + scene_id + '.npy'
            if os.path.exists(plane_file):
                newList.append(self.dataList[i])
        self.dataList = newList
    print(len(self.dataList))
    #import pdb; pdb.set_trace()
    self.dataList = self.dataList
    self.len = len(self.dataList)

    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    self.Rs = Rs
    self.sift = cv2.xfeatures2d.SIFT_create()
    
  def Pano2PointCloud(self,depth,representation):
    if representation == 'skybox':
      assert(depth.shape[0]==160 and depth.shape[1]==640)
      if depth.shape[0]==160 and depth.shape[1]==640:
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        masks=[]
        for i in range(4):
          zs = depth[:,i*w:(i+1)*w].flatten()
          mask=(zs!=0)
          zs=zs[mask]
          ys_this, xs_this = ys.flatten()[mask]*zs, xs.flatten()[mask]*zs
          pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
          pc_this = np.matmul(self.Rs[(i-1)%4][:3,:3],pc_this)
          pc.append(pc_this)
          masks.append(np.where(mask)[0]+h*h*i)
        pc = np.concatenate(pc,1)
        masks=np.concatenate(masks)
    else:
      raise Exception("unknown representation")
    return pc,masks

  def PanoIdx(self,index,h,w,representation):
    if representation == 'skybox':
      total=h*w
      single=total//4
      hidx = index//single
      rest=index % single
      ys,xs=np.unravel_index(rest, [h,h])
      xs += hidx*h
      idx = np.zeros([len(xs),2])
      idx[:,0]=xs
      idx[:,1]=ys
    return idx

  def __getpair__(self, index):
    self.base_this = self.dataList[index]['base']
    self.interval_this = '0-15'
    ct0,ct1=self.dataList[index]['id_src'],self.dataList[index]['id_tgt']
    return ct0,ct1

  def LoadImage(self, PATH,depth=True):
    try:
        if depth:
          img = cv2.imread(PATH,2)/1000.
        else:
          img = cv2.imread(PATH)
    except:
        import pdb; pdb.set_trace()
    return img
  
  def shuffle(self):
    pass
  
  def reproj_helper(self,pct,colorpct,out_shape,mode):
      # find which plane they intersect with
      h=out_shape[0]
      tp=np.matmul(self.Rs[3][:3,:3].T,pct)
      tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
      intersectf=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)
      if mode in ['color','normal']:
        colorf=colorpct[intersectf,:]
      elif mode == 'depth':
        colorf=-tp[2,intersectf]
      coordf=tp[:2,intersectf]
      coordf[0,:]=(coordf[0,:]+1)*0.5*h
      coordf[1,:]=(1-coordf[1,:])*0.5*h
      coordf=coordf.round().clip(0,h-1).astype('int')

      tp=np.matmul(self.Rs[0][:3,:3].T,pct)
      tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
      intersectr=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

      if mode in ['color','normal']:
        colorr=colorpct[intersectr,:]
      elif mode == 'depth':
        colorr=-tp[2,intersectr]

      coordr=tp[:2,intersectr]
      coordr[0,:]=(coordr[0,:]+1)*0.5*h
      coordr[1,:]=(1-coordr[1,:])*0.5*h
      coordr=coordr.round().clip(0,h-1).astype('int')
      coordr[0,:]+=h

      tp=np.matmul(self.Rs[1][:3,:3].T,pct)
      tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
      intersectb=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

      if mode in ['color','normal']:
        colorb=colorpct[intersectb,:]
      elif mode == 'depth':
        colorb=-tp[2,intersectb]

      coordb=tp[:2,intersectb]
      coordb[0,:]=(coordb[0,:]+1)*0.5*h
      coordb[1,:]=(1-coordb[1,:])*0.5*h
      coordb=coordb.round().clip(0,h-1).astype('int')
      coordb[0,:]+=h*2

      tp=np.matmul(self.Rs[2][:3,:3].T,pct)
      tp[:2,:]/=(np.abs(tp[2,:])+1e-32)
      intersectl=(tp[2,:]<0)*(np.abs(tp[0,:])<1)*(np.abs(tp[1,:])<1)

      if mode in ['color','normal']:
        colorl=colorpct[intersectl,:]
      elif mode == 'depth':
        colorl=-tp[2,intersectl]

      coordl=tp[:2,intersectl]
      coordl[0,:]=(coordl[0,:]+1)*0.5*h
      coordl[1,:]=(1-coordl[1,:])*0.5*h
      coordl=coordl.round().clip(0,h-1).astype('int')
      coordl[0,:]+=h*3

      proj=np.zeros(out_shape)

      proj[coordf[1,:],coordf[0,:]]=colorf
      proj[coordl[1,:],coordl[0,:]]=colorl
      proj[coordb[1,:],coordb[0,:]]=colorb
      proj[coordr[1,:],coordr[0,:]]=colorr
      return proj

  def depth2pc(self,depth, need_mask=False):
      if (depth.shape[0] == 480 and depth.shape[1] == 640):
        w,h = depth.shape[1], depth.shape[0]
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        xs=xs.flatten()[mask]*zs/(0.8921875*2)
        ys=ys.flatten()[mask]*zs/(1.1895*2)
        pc = np.stack((xs,ys,-zs),1)
      elif (depth.shape[0] == 320 and depth.shape[1] == 640):
        w,h = depth.shape[1], depth.shape[0]
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        hfov = 120.0
        vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*320/640)/np.pi*180
        xs=xs.flatten()[mask]*zs*(np.tan(np.deg2rad(hfov/2)))
        ys=ys.flatten()[mask]*zs*(np.tan(np.deg2rad(vfov/2)))
        pc = np.stack((xs,ys,-zs),1)
      elif (depth.shape[0] == 200 and depth.shape[1] == 400):
        w,h = depth.shape[1], depth.shape[0]
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        hfov = 120.0
        vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*200/400)/np.pi*180
        xs=xs.flatten()[mask]*zs*(np.tan(np.deg2rad(hfov/2)))
        ys=ys.flatten()[mask]*zs*(np.tan(np.deg2rad(vfov/2)))
        pc = np.stack((xs,ys,-zs),1)
      elif (depth.shape[0] == 160 and depth.shape[1] == 160):
        w,h = depth.shape[1], depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        ys, xs = ys.flatten()*zs, xs.flatten()*zs
        mask = (zs!=0)
        pc = np.concatenate((xs[mask],ys[mask],-zs[mask])).reshape(3,-1).T
      elif (depth.shape[0] == 154 and depth.shape[1] == 206):
        w,h = depth.shape[1], depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        hfov = 120.0
        vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*320/640)/np.pi*180
        ys, xs = ys.flatten()*zs*(np.tan(np.deg2rad(vfov/2))), xs.flatten()*zs*(np.tan(np.deg2rad(hfov/2)))
        mask = (zs!=0)
        pc = np.concatenate((xs[mask]/640*w,ys[mask]/320*h,-zs[mask])).reshape(3,-1).T
      elif (depth.shape[0] == 96 and depth.shape[1] == 128):
        w,h = depth.shape[1], depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        zs = depth.flatten()
        hfov = 120.0
        vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*200/400)/np.pi*180
        ys, xs = ys.flatten()*zs*(np.tan(np.deg2rad(vfov/2))), xs.flatten()*zs*(np.tan(np.deg2rad(hfov/2)))
        mask = (zs!=0)
        pc = np.concatenate((xs[mask]/400*w,ys[mask]/200*h,-zs[mask])).reshape(3,-1).T
      
      if need_mask:
        return pc,mask
      else:
        return pc

  def __getitem__(self, index):
    rets = {}
    imgs = np.zeros((self.nViews, *self.OutputSize[::-1]), dtype = np.float32)
    if self.rgbd:
      imgs_rgb = np.zeros((self.nViews, *self.OutputSize[::-1], 3), dtype = np.float32)
    if self.segm:
      segm = np.zeros((self.nViews,1,*self.OutputSize[::-1]), dtype = np.float32)
      if self.dynamicWeighting:
        dynamicW = np.zeros((self.nViews,1,*self.OutputSize[::-1]), dtype = np.float32)
    if self.normal:
      normal = np.zeros((self.nViews,*self.OutputSize[::-1],3), dtype = np.float32)
    if self.pointcloud:
      pointcloud = np.zeros((self.nViews, 3+3+3+1, self.num_points), dtype = np.float32)
      pointcloud_flow = np.zeros((self.nViews, 3, self.num_points), dtype = np.float32)
    
    R = np.zeros((self.nViews, 4, 4))
    Q = np.zeros((self.nViews, 7))
    assert(self.nViews == 2)
    ct0,ct1 = self.__getpair__(index)
    imgsPath = []
    basePath = self.base_this
    frameid0 = f"{ct0:06d}"
    frameid1 = f"{ct1:06d}"
    

    if self.fullsize_rgbdn:
      imgs_rgb_full = np.zeros((self.nViews, 480,640, 3), dtype = np.float32)
      imgs_full = np.zeros((self.nViews, 480,640), dtype = np.float32)
      imgs_full[0] = self.LoadImage(os.path.join(basePath,'obs_depth','{}.png'.format(frameid0))).copy()
      imgs_full[1] = self.LoadImage(os.path.join(basePath,'obs_depth','{}.png'.format(frameid1))).copy()
      imgs_rgb_full[0] = self.LoadImage(os.path.join(basePath,'obs_rgb','{}.png'.format(frameid0)),depth=False).copy()/255.
      imgs_rgb_full[1] = self.LoadImage(os.path.join(basePath,'obs_rgb','{}.png'.format(frameid1)),depth=False).copy()/255.
      rets['rgb_full'] = imgs_rgb_full[np.newaxis,:]
      rets['depth_full'] = imgs_full[np.newaxis,:]
    
    imgs[0] = self.LoadImage(os.path.join(basePath,'depth','{}.png'.format(frameid0))).copy()
    imgs[1] = self.LoadImage(os.path.join(basePath,'depth','{}.png'.format(frameid1))).copy()
    dataMask = np.zeros((self.nViews, 1,*self.OutputSize[::-1]), dtype = np.float32)
    dataMask[0,0,:,:]=(imgs[0]!=0)
    dataMask[1,0,:,:]=(imgs[1]!=0)
    rets['dataMask']=dataMask[np.newaxis,:]

    if self.rgbd:
      imgs_rgb[0] = self.LoadImage(os.path.join(basePath,'rgb','{}.png'.format(frameid0)),depth=False).copy()/255.
      imgs_rgb[1] = self.LoadImage(os.path.join(basePath,'rgb','{}.png'.format(frameid1)),depth=False).copy()/255.
    
    if self.scannet_new_name:
        tmp_basePath = basePath.replace('ScanNet_360','ScanNet')
    else:
        tmp_basePath = basePath

    R[0] = np.loadtxt(os.path.join(tmp_basePath,'pose', frameid0 + '.pose.txt'))
    R[1] = np.loadtxt(os.path.join(tmp_basePath,'pose', frameid1 + '.pose.txt'))
    Q[0,:4] = rot2Quaternion(R[0][:3,:3])
    Q[0,4:] = R[0][:3,3]
    Q[1,:4] = rot2Quaternion(R[1][:3,:3])
    Q[1,4:] = R[1][:3,3]
    imgsPath.append(f"{basePath}/{ct0:06d}")
    imgsPath.append(f"{basePath}/{ct1:06d}")

    if self.normal:
      tp=self.LoadImage(os.path.join(basePath,'normal','{}.png'.format(frameid0)),depth=False).copy().astype('float')
      mask=(tp==0).sum(2)<3
      tp[mask]=tp[mask]/255.*2-1
      normal[0]=tp
      tp=self.LoadImage(os.path.join(basePath,'normal','{}.png'.format(frameid1)),depth=False).copy().astype('float')
      mask=(tp==0).sum(2)<3
      tp[mask]=tp[mask]/255.*2-1
      normal[1]=tp
    
    if self.segm:
      tp = (self.LoadImage(os.path.join(basePath,'semantic_idx','{}.png'.format(frameid0)),depth=False).copy())[:,:,1]
      segm[0] = tp.reshape(segm[0].shape)
      tp = (self.LoadImage(os.path.join(basePath,'semantic_idx','{}.png'.format(frameid1)),depth=False).copy())[:,:,1]

      segm[1] = tp.reshape(segm[1].shape)

      segm_ = np.zeros((self.nViews,1,*self.OutputSize[::-1]), dtype = np.float32)
      segm_[0] = segm[0]
      segm_[1] = segm[1]
      segm_ = segm_[np.newaxis,:]

    if self.denseCorres:
        # get 3d point cloud for each pano
        pcs,masks = self.Pano2PointCloud(imgs[0],self.representation) # be aware of the order of returned pc!!!
        pct,maskt = self.Pano2PointCloud(imgs[1],self.representation)
        #import pdb; pdb.set_trace()
        #pct = np.matmul(R[0],np.matmul(np.linalg.inv(R[1]),np.concatenate((pct,np.ones([1,pct.shape[1]])))))[:3,:]
        pct = np.matmul(np.linalg.inv(R[1]),np.concatenate((pct,np.ones([1,pct.shape[1]]))))[:3,:]
        pcs = np.matmul(np.linalg.inv(R[0]),np.concatenate((pcs,np.ones([1,pcs.shape[1]]))))[:3,:]
        # find correspondence using kdtree
        tree = KDTree(pct.T)
        IdxQuery=np.random.choice(range(pcs.shape[1]),5000)
        # sample 5000 query points
        pcsQuery = pcs[:,IdxQuery]
        nearest_dist, nearest_ind = tree.query(pcsQuery.T, k=1)
        hasCorres=(nearest_dist < 0.08)
        idxTgtNeg=[]
        idxSrc=self.PanoIdx(masks[IdxQuery[np.where(hasCorres)[0]]],imgs.shape[1],imgs.shape[2],self.representation)
        idxTgt=self.PanoIdx(maskt[nearest_ind[hasCorres]],imgs.shape[1],imgs.shape[2],self.representation)

        if hasCorres.sum() < 200:
          rets['denseCorres']={'idxSrc':np.zeros([1,500,2]),'idxTgt':np.zeros([1,500,2]),'valid':np.array([0]),'idxTgtNeg':idxTgtNeg}

        else:
          # only pick 2000 correspondence per pair
          idx500 = np.random.choice(range(idxSrc.shape[0]),500)
          idxSrc=idxSrc[idx500][np.newaxis,:]
          idxTgt=idxTgt[idx500][np.newaxis,:]

          rets['denseCorres']={'idxSrc':idxSrc,'idxTgt':idxTgt,'valid':np.array([1]),'idxTgtNeg':idxTgtNeg}
    
    imgPCid = np.zeros([2, self.num_points, 2])

    if self.pointcloud:
        try:
            pc = self.depth2pc(imgs[0][:,160:160*2])

            idx_s = np.random.choice(range(len(pc)),self.num_points)

            imgPCid[0] = np.stack((idx_s % 160, idx_s // 160)).T
            pointcloud[0,:3,:] = pc[idx_s,:].T

            pc_n = normal[0][:,160:160*2, :].reshape(-1, 3)
            pointcloud[0,3:6,:] = pc_n[idx_s,:].T

            pc_c = imgs_rgb[0,:,160:160*2,:].reshape(-1,3)

            pointcloud[0,6:9,:] = pc_c[idx_s,::-1].T

            #pc_s = imgs_s[0,:,160:160*2].reshape(-1)+1
            #pointcloud[0,9:10,:] = pc_s[idx_s]


            pc = self.depth2pc(imgs[1][:,160:160*2])
            idx_s = np.random.choice(range(len(pc)),self.num_points)
            imgPCid[1] = np.stack((idx_s % 160, idx_s // 160)).T
            pointcloud[1,:3,:] = pc[idx_s,:].T

            pc_n = normal[1][:,160:160*2,:].reshape(-1, 3)
            pointcloud[1,3:6,:] = pc_n[idx_s,:].T

            pc_c = imgs_rgb[1,:,160:160*2,:].reshape(-1,3)

            #pc_s = imgs_s[1,:, 160:160*2].reshape(-1)+1
            #pointcloud[1,9:10,:] = pc_s[idx_s]
        except:
            #import pdb; pdb.set_trace()
            pointcloud = np.zeros((self.nViews, 3+3+3+1, self.num_points), dtype = np.float32)
            pointcloud_flow = np.zeros((self.nViews, 3, self.num_points), dtype = np.float32)
            print("this pair does not contain point cloud!")
    if self.plane_r:

        scene_id = basePath.split('/')[-1]

        plane_file = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/ScanNet_plane/train/' + scene_id + '.npy'
        if os.path.exists(plane_file):
            plane_eq_raw = np.load(plane_file)
            if plane_eq_raw.shape[0] < 6:
                plane_eq_raw = np.concatenate([plane_eq_raw,plane_eq_raw],axis=0)
            MAX_PLANE = 10
            plane_idx = np.argsort(plane_eq_raw[:,7])

            plane_eq_raw = plane_eq_raw[plane_idx[-MAX_PLANE:]]
            truncate_num = plane_eq_raw[-6,7] / 2
            plane_eq_raw = plane_eq_raw[plane_eq_raw[:,7] > truncate_num]


            if plane_eq_raw.shape[0] < MAX_PLANE:
                valid_plane = plane_eq_raw.shape[0]
                plane_eq_raw = np.concatenate((plane_eq_raw, np.zeros([MAX_PLANE - plane_eq_raw.shape[0], plane_eq_raw.shape[-1]])))
            else:
                valid_plane = MAX_PLANE

            plane_eq = plane_eq_raw[:,3:7]
            plane_eq = np.matmul(plane_eq, np.linalg.inv(R[0]))
            plane_center = plane_eq_raw[:,:3]
            plane_center = (np.matmul(R[0][:3,:3], plane_center.T) + R[0][:3,3:4]).T
            
            #import pdb; pdb.set_trace()
        else:
            print("Missing plane data")
            import pdb; pdb.set_trace()
        
    
    if self.plane_m:
        scene_id = basePath.split('/')[-1]

        plane_file = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/ScanNet_manual_plane/%s/' % self.split + scene_id + '.npy'

        plane_raw = np.load(plane_file,allow_pickle=True)

        plane_center = plane_raw[:,:3]
        plane_center = (np.matmul(R[0][:3,:3],plane_center.T)+R[0][:3,3:4]).T

        plane_normal = plane_raw[:,3:6]
        #plane_normal = (np.matmul(R[0][:3,:3],plane_normal.T)+R[0][:3,3:4]).T
        plane_normal = np.matmul(plane_normal, np.linalg.inv(R[0][:3,:3]))

        rets['plane_c'] = plane_center[np.newaxis,:]
        rets['plane_n'] = plane_normal[np.newaxis,:]
        rets['plane_raw'] = plane_raw[np.newaxis,:]

    # reprojct the second image into the first image plane
    if self.reproj:
      
      assert(imgs.shape[1]==160 and imgs.shape[2]==640)
      h=imgs.shape[1]

      pct,mask = util.depth2pc(imgs[1,80-33:80+33,160+80-44:160+80+44],'scannet')# be aware of the order of returned pc!!!

      colorpct = imgs_rgb[1,80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
      normalpct = normal[1,80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
      depthpct = imgs[1,80-33:80+33,160+80-44:160+80+44].reshape(-1)[mask]

      R_this=np.matmul(R[0],np.linalg.inv(R[1]))
      R_this_p=R_this.copy()
      dR=util.randomRotation(epsilon=0.1)
      dRangle=angular_distance_np(dR[np.newaxis,:],np.eye(3)[np.newaxis,:])[0]
      
      R_this_p[:3,:3]=np.matmul(dR,R_this_p[:3,:3])
      R_this_p[:3,3]+=np.random.randn(3)*0.1

      t2s_dr = np.matmul(R_this, np.linalg.inv(R_this_p))
      
      pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
      pct_reproj_org = np.matmul(R_this,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
      flow = pct_reproj_org - pct_reproj
      normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
      flow = flow.T
      t2s_rgb=self.reproj_helper(pct_reproj_org,colorpct,imgs_rgb[0].shape,'color')
      t2s_rgb_p=self.reproj_helper(pct_reproj,colorpct,imgs_rgb[0].shape,'color')
      t2s_n_p=self.reproj_helper(pct_reproj,normalpct,imgs_rgb[0].shape,'normal')
      t2s_d_p=self.reproj_helper(pct_reproj,depthpct,imgs_rgb[0].shape[:2],'depth')
      
      t2s_flow_p=self.reproj_helper(pct_reproj,flow,imgs_rgb[0].shape,'color')
      t2s_mask_p=(t2s_d_p!=0).astype('int')


      pct,mask = util.depth2pc(imgs[0,80-33:80+33,160+80-44:160+80+44],'scannet')# be aware of the order of returned pc!!!

      colorpct = imgs_rgb[0,80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
      normalpct = normal[0,80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
      depthpct = imgs[0,80-33:80+33,160+80-44:160+80+44].reshape(-1)[mask]

      R_this=np.matmul(R[1],np.linalg.inv(R[0]))
      R_this_p=R_this.copy()
      dR=util.randomRotation(epsilon=0.1)
      dRangle=angular_distance_np(dR[np.newaxis,:],np.eye(3)[np.newaxis,:])[0]
      
      R_this_p[:3,:3]=np.matmul(dR,R_this_p[:3,:3])
      R_this_p[:3,3]+=np.random.randn(3)*0.1
      s2t_dr = np.matmul(R_this, np.linalg.inv(R_this_p))
      pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
      pct_reproj_org = np.matmul(R_this,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
      flow = pct_reproj_org - pct_reproj
      # assume always observe the second view(right view)
      
      normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
      flow = flow.T

      s2t_rgb=self.reproj_helper(pct_reproj_org,colorpct,imgs_rgb[0].shape,'color')
      s2t_rgb_p=self.reproj_helper(pct_reproj,colorpct,imgs_rgb[0].shape,'color')
      s2t_n_p=self.reproj_helper(pct_reproj,normalpct,imgs_rgb[0].shape,'normal')
      s2t_d_p=self.reproj_helper(pct_reproj,depthpct,imgs_rgb[0].shape[:2],'depth')
      s2t_flow_p=self.reproj_helper(pct_reproj,flow,imgs_rgb[0].shape,'color')
      s2t_mask_p=(s2t_d_p!=0).astype('int')

      # compute an envelop box
      try:
        tp=np.where(t2s_d_p.sum(0))[0]
        w0,w1=tp[0],tp[-1]
        tp=np.where(t2s_d_p.sum(1))[0]
        h0,h1=tp[0],tp[-1]
      except:
        w0,h0=0,0
        w1,h1=t2s_d_p.shape[1]-1,t2s_d_p.shape[0]-1
      t2s_box_p = np.zeros(t2s_d_p.shape)
      t2s_box_p[h0:h1,w0:w1] = 1

      try:
        tp=np.where(s2t_d_p.sum(0))[0]
        w0,w1=tp[0],tp[-1]
        tp=np.where(s2t_d_p.sum(1))[0]
        h0,h1=tp[0],tp[-1]
      except:
        w0,h0=0,0
        w1,h1=s2t_d_p.shape[1]-1,s2t_d_p.shape[0]-1
      s2t_box_p = np.zeros(s2t_d_p.shape)
      s2t_box_p[h0:h1,w0:w1] = 1

      rets['proj_dr'] = np.stack((t2s_dr,s2t_dr),0)[np.newaxis,:]
      rets['proj_flow'] =np.stack((t2s_flow_p,s2t_flow_p),0).transpose(0,3,1,2)[np.newaxis,:]
      rets['proj_rgb'] =np.stack((t2s_rgb,s2t_rgb),0).transpose(0,3,1,2)[np.newaxis,:]
      rets['proj_rgb_p'] =np.stack((t2s_rgb_p,s2t_rgb_p),0).transpose(0,3,1,2)[np.newaxis,:]
      rets['proj_n_p']   =np.stack((t2s_n_p,s2t_n_p),0).transpose(0,3,1,2)[np.newaxis,:]
      rets['proj_d_p']   =np.stack((t2s_d_p,s2t_d_p),0).reshape(1,2,1,t2s_d_p.shape[0],t2s_d_p.shape[1])
      rets['proj_mask_p']=np.stack((t2s_mask_p,s2t_mask_p),0).reshape(1,2,1,t2s_mask_p.shape[0],t2s_mask_p.shape[1])
      rets['proj_box_p'] = np.stack((t2s_box_p,s2t_box_p),0).reshape(1,2,1,t2s_box_p.shape[0],t2s_box_p.shape[1])

    


    imgs = imgs[np.newaxis,:]
    if self.rgbd:
      imgs_rgb = imgs_rgb[np.newaxis,:].transpose(0,1,4,2,3)
    if self.normal:
      normal = normal[np.newaxis,:].transpose(0,1,4,2,3)
    R = R[np.newaxis,:]
    Q = Q[np.newaxis,:]
    if self.segm:
      rets['segm']=segm_
      if self.dynamicWeighting:
        rets['dynamicW'] = dynamicW[np.newaxis,:]

    if self.pointcloud:
        pointcloud = pointcloud[np.newaxis,:]
        pointcloud_flow = pointcloud_flow[np.newaxis, :]
        rets['pointcloud']=pointcloud
        rets['pointcloud_flow']=pointcloud_flow
        
    if self.plane_r:
        rets['plane']=plane_eq[np.newaxis,:]
        rets['plane_raw']=plane_eq_raw[np.newaxis,:]
        rets['plane_c']=plane_center[np.newaxis,:]
        rets['valid_plane']=valid_plane

    rets['interval']=self.interval_this
    rets['norm']=normal
    rets['rgb']=imgs_rgb
    rets['depth']=imgs
    rets['Q']=Q
    rets['R']=R
    rets['imgsPath']=imgsPath
    return rets
    
  def __len__(self):
    return self.len


