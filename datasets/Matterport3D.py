import torch.utils.data as data
import numpy as np
import torch
import cv2
import config
import os
import glob
import sys
sys.path.append("../")
from util import rot2Quaternion,angular_distance_np
import util
import scipy.io as sio
from scipy import ndimage
from sklearn.neighbors import KDTree

class Matterport3D(data.Dataset):
  def __init__(self, split, nViews, imgSize=224, AuthenticdepthMap=False, crop=False, cache=True,\
        hmap=False,CorresCoords=False,meta=False,rotate=False,rgbd=False,birdview=False,pointcloud=False,num_points=8192,\
        denseCorres=False,segm=False,eval_local=False,local_eval_list=None, local=False,objectCloud=False,filter_overlap=False,topdown=False,reproj=False,singleView=True,dynamicWeighting=False,normal=False,list_=None, corner=False, plane=False, plane_r=False, plane_m=False, small_debug=0, scannet_new_name=0, representation='skybox',entrySplit=None,snumclass=0):
    self.crop = crop
    self.objectCloud = objectCloud
    self.filter_overlap = filter_overlap
    self.pointcloud = pointcloud
    self.birdview = birdview
    self.num_points = num_points
    self.rgbd = rgbd
    self.rotate = rotate
    self.meta = meta
    self.local = local
    self.AuthenticdepthMap = AuthenticdepthMap
    self.hmap = hmap
    self.CorresCoords = CorresCoords
    self.split = split
    self.nViews = nViews
    self.imgSize = imgSize
    self.normal = normal
    self.reproj = reproj
    self.singleView = singleView
    self.topdown = topdown
    self.list=list_
    self.denseCorres=denseCorres
    self.representation = representation
    self.entrySplit=entrySplit
    self.segm = segm
    self.dynamicWeighting = dynamicWeighting
    self.plane_r = plane_r
    self.plane_m = plane_m
    self.local_eval_list = local_eval_list

    if self.dynamicWeighting:
      assert(self.segm == True)
    self.OutputSize = (640,160)
    self.Inputwidth = config.pano_width
    self.Inputheight = config.pano_height
    self.nPanoView = 4
    self.intrinsic = np.array([[571.623718/640,0,319.500000/640],[0,571.623718/480,239.500000/480],[0,0,1]])
    self.intrinsicUnNorm = np.array([[571.623718,0,319.500000],[0,571.623718,239.500000],[0,0,1]])
    self.snumclass = snumclass
    self.dataList = np.load(self.list, allow_pickle=True).item()[self.split]
    self.eval_local =eval_local

    #dataS=sio.loadmat('data/test_data/matterport_source_clean_v1_top5.mat')
    
    #dataT=sio.loadmat('data/test_data/matterport_target_clean_v1_top5.mat')
    
    if self.eval_local:
      self.eval_gt_dict = {}
      new_list=[]
      if 1:
        list_local = np.load(self.local_eval_list, allow_pickle=True)
        for i in range(len(list_local)):
          room_id = list_local[i][0].split('-')[1]
          scene_id = list_local[i][0].split('-')[0]

          id_src = int(list_local[i][1])
          id_tgt = int(list_local[i][2])
          
          gt_pose = list_local[i][4]
          for j in range(1):
            
            pred_pose = list_local[i][3][j]
            # pred_pose = list_local[i][3]
            # pred_pose = np.eye(4)
            if os.path.exists('/scratch'):
                base = '/scratch/cluster/yzp12/projects/2019_ICLR_RelativePose/data_ssd/Matterport3D/imagev2/' + '/test/' + scene_id + '/' + room_id
            else:
                base = '/home/yzp12/projects/2019_ICLR_RelativePose/data_ssd/Matterport3D/imagev2/' + '/test/' + scene_id + '/' + room_id
                # base = self.dataList[0]['base'].split('test')[0]+'/test/' + scene_id + '/' + room_id
            new_list.append({'base':base,
            'id_src':id_src,
            'id_tgt':id_tgt,
            'Kth':j,
            'overlap':list_local[i][5],
            })
            self.eval_gt_dict['%s-%06d-%06d-%d' % (scene_id+'-'+room_id, id_src, id_tgt, j)] = {'pred_pose':pred_pose,
                                                                              'gt_pose':gt_pose,
                                                                              'pos_s_360':np.zeros([1,3]),
                                                                              'pos_t_360':np.zeros([1,3]),
                                                                              'nor_s_360':np.zeros([1,3]),
                                                                              'nor_t_360':np.zeros([1,3]),
                                                                              'feat_s_360':np.zeros([1,3]),
                                                                              'feat_t_360':np.zeros([1,3]),
                                                                              }

        self.dataList = new_list
    if 0:
      new_list = []
      for i in range(len(dataS['path'])):
        tp=dataS['path'][i,0]
        tp1=dataT['path'][i,0]
        new_list.append({'base':'/home/yzp12/projects/2019_ICLR_RelativePose/data_ssd/Matterport3D/imagev2/test/%s/%s' % (tp.split('/')[-3], tp.split('/')[-2]),
        'id_src':int(tp.split('/')[-1]), 
        'id_tgt':int(tp1.split('/')[-1]),'overlap':0.0 })
      
      self.dataList = new_list
    
    print("before datalist:", len(self.dataList))
    #import pdb; pdb.set_trace() 
    if self.plane_m:
        #import pdb; pdb.set_trace()
        newList = []
        for i in range(len(self.dataList)):
            tp = self.dataList[i]['base']
            scene_id = tp.split('%s/' % self.split)[-1]
            #if 'scene0271_01' not in scene_id:
            #    continue
            prefix = '/home/yzp12/projects/2020_CVPR_Hybrid/data/siming/'
            if os.path.exists(os.path.join(prefix, 'Matterport3D_PCD/imagev2', self.split, scene_id, 'plane.npy')):
            #if os.path.exists(os.path.join(prefix, 'ScanNet_manual_plane', self.split, scene_id+'.npy')):
              if os.path.exists(os.path.join(prefix, 'Matterport3D_PCD/imagev2', self.split, scene_id, 'plane_idx', '%06d.png'%(self.dataList[i]['id_src']))) and \
                os.path.exists(os.path.join(prefix, 'Matterport3D_PCD/imagev2', self.split, scene_id, 'plane_idx', '%06d.png'%(self.dataList[i]['id_tgt']))):
                newList.append(self.dataList[i])

        self.dataList = newList  
    
    print("after datalist:", len(self.dataList))

    if self.entrySplit is not None:
        self.dataList = [self.dataList[kk] for kk in range(self.entrySplit*100,(self.entrySplit+1)*100)]
    self.len = len(self.dataList)

    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    self.Rs = Rs
    self.sift = cv2.xfeatures2d.SIFT_create()
    
  def Pano2PointCloud(self,depth):
    assert(depth.shape[0]==160 and depth.shape[1]==640)
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
    return pc,masks

  def PanoIdx(self,index,h,w):
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
    self.base_this = self.dataList[index]['base'].replace('data_ssd1','data_ssd')
    self.interval_this = '0-15'
    ct0,ct1=self.dataList[index]['id_src'],self.dataList[index]['id_tgt']
    return ct0,ct1

  def LoadImage(self, PATH,depth=True):

    if depth:
      img = cv2.imread(PATH,2)/1000.
    else:
      img = cv2.imread(PATH)
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
    
  def __getitem__(self, index):
    #import ipdb;ipdb.set_trace()
    rets = {}
    imgs_ = np.zeros((self.nViews, *self.OutputSize[::-1]), dtype = np.float32)
    imgs = np.zeros((self.nViews, self.Inputheight, self.Inputwidth), dtype = np.float32)
    if self.rgbd:
      imgs_rgb = np.zeros((self.nViews, self.Inputheight, self.Inputwidth,3), dtype = np.float32)
      imgs_rgb_ = np.zeros((self.nViews,3,*self.OutputSize[::-1]), dtype = np.float32)
    if self.segm:
      segm = np.zeros((self.nViews,1,*self.OutputSize[::-1]), dtype = np.float32)
    if self.normal:
      normal = np.zeros((self.nViews,3,self.Inputheight,self.Inputwidth), dtype = np.float32)

    pointcloud = np.zeros((self.nViews, 3+3+3+1, self.num_points), dtype = np.float32)
    R = np.zeros((self.nViews, 4, 4))
    Q = np.zeros((self.nViews, 7))
    assert(self.nViews == 2)
    ct0,ct1 = self.__getpair__(index)
    imgsPath = []
    basePath = self.base_this
    
    frameid0 = f"{ct0:06d}"
    frameid1 = f"{ct1:06d}"
    rets['overlap'] = float(self.dataList[index]['overlap'])
    
    scene_id = basePath.split('/')[-2]
    room_id = scene_id + '-' + basePath.split('/')[-1]
    
    
    imgs[0] = self.LoadImage(os.path.join(basePath,'depth','{}.png'.format(frameid0))).copy()
    imgs[1] = self.LoadImage(os.path.join(basePath,'depth','{}.png'.format(frameid1))).copy()
    PerspectiveValidMask = (imgs!=0)
    rets['PerspectiveValidMask'] = PerspectiveValidMask[None,:,None,:,:]
    rets['dataMask'] = rets['PerspectiveValidMask']
    
    dataMask = np.zeros((self.nViews, 1,*self.OutputSize[::-1]), dtype = np.float32)
    dataMask[0,0,:,:]=(imgs[0]!=0)
    dataMask[1,0,:,:]=(imgs[1]!=0)
    rets['dataMask']=dataMask[np.newaxis,:]
    if self.rgbd:
      imgs_rgb[0] = self.LoadImage(os.path.join(basePath,'rgb','{}.png'.format(frameid0)),depth=False).copy()/255.
      imgs_rgb[1] = self.LoadImage(os.path.join(basePath,'rgb','{}.png'.format(frameid1)),depth=False).copy()/255.
    R[0] = np.loadtxt(os.path.join(basePath,'pose', frameid0 + '.pose.txt'))
    R[1] = np.loadtxt(os.path.join(basePath,'pose', frameid1 + '.pose.txt'))
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
      normal[0]=tp.transpose(2,0,1)
      tp=self.LoadImage(os.path.join(basePath,'normal','{}.png'.format(frameid1)),depth=False).copy().astype('float')
      mask=(tp==0).sum(2)<3
      tp[mask]=tp[mask]/255.*2-1
      normal[1]=tp.transpose(2,0,1)

      normal_ = np.zeros((self.nViews,3,*self.OutputSize[::-1]), dtype = np.float32)
      normal_[0] = cv2.resize(normal[0].transpose(1,2,0),self.OutputSize,interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
      normal_[1] = cv2.resize(normal[1].transpose(1,2,0),self.OutputSize,interpolation=cv2.INTER_NEAREST).transpose(2,0,1)
      normal_ = normal_[np.newaxis,:]
    
    if 1:
      segm = np.zeros((self.nViews,1,*self.OutputSize[::-1]), dtype = np.float32)
      tp = (self.LoadImage(os.path.join(basePath,'semanticLabel','{}.png'.format(frameid0)),depth=False)[:,:,0].copy())
      segm[0] = tp.reshape(segm[0].shape)
      tp = (self.LoadImage(os.path.join(basePath,'semanticLabel','{}.png'.format(frameid1)),depth=False)[:,:,0].copy())
      segm[1] = tp.reshape(segm[1].shape)
      segm[0] = segm[0]
      segm[1] = segm[1]
      # truncate semantic class
      segm[segm>=self.snumclass]=0
      segm = segm[np.newaxis,:]
      segm = np.squeeze(segm, 2)

    if self.denseCorres:
        # get 3d point cloud for each pano
        
        pcs,masks = self.Pano2PointCloud(imgs[0]) # be aware of the order of returned pc!!!
        pct,maskt = self.Pano2PointCloud(imgs[1])

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
        
        idxSrc=self.PanoIdx(masks[IdxQuery[np.where(hasCorres)[0]]],160,640)
        idxTgt=self.PanoIdx(maskt[nearest_ind[hasCorres]],160,640)
        
        if hasCorres.sum() < 500:
          rets['denseCorres']={'idxSrc':np.zeros([1,2000,2]),'idxTgt':np.zeros([1,2000,2]),'valid':np.array([0]),'idxTgtNeg':idxTgtNeg}

        else:
          # only pick 2000 correspondence per pair
          idx2000 = np.random.choice(range(idxSrc.shape[0]),2000)
          idxSrc=idxSrc[idx2000][np.newaxis,:]
          idxTgt=idxTgt[idx2000][np.newaxis,:]

          rets['denseCorres']={'idxSrc':idxSrc,'idxTgt':idxTgt,'valid':np.array([1]),'idxTgtNeg':idxTgtNeg}
    
    imgPCid = np.zeros([2, self.num_points, 2])
    R_inv = np.linalg.inv(R)

    if self.pointcloud or self.local:
        
        #pc = self.depth2pc(imgs_depth[0][:,160:160*2]).T
        pc, mask = util.depth2pc(imgs[0][:,160:160*2], 'matterport')
        
        # util.write_ply('test.ply',np.concatenate((pc,pc1)))
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        mask_s = np.where(mask)[0][idx_s]
        # imgPCid[0] = np.stack((idx_s % 160, idx_s // 160)).T
        imgPCid[0] = np.stack((idx_s % 128, idx_s // 128)).T
        pointcloud[0,:3,:] = pc[idx_s,:].T
        # pc_n = imgs_normal[0][:,160:160*2].reshape(-1, 3)
        pc_n = normal[0].transpose(1,2,0)[:,160:160*2, :].reshape(-1, 3)[mask]
        pointcloud[0,3:6,:] = pc_n[idx_s,:].T
        # pc_c = imgs_rgb[0,:,160:160*2,:].reshape(-1,3)
        pc_c = imgs_rgb[0,:,160:160*2,:].reshape(-1,3)[mask]
        pointcloud[0,6:9,:] = pc_c[idx_s,::-1].T
        # pc_s = imgs_s[0,:,160:160*2].reshape(-1)


        # pc = self.depth2pc(imgs_depth[1][:,160:160*2]).T
        pc, mask = util.depth2pc(imgs[1][:,160:160*2], 'matterport')
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        mask_t = np.where(mask)[0][idx_s]
        # imgPCid[1] = np.stack((idx_s % 160, idx_s // 160)).T
        imgPCid[1] = np.stack((idx_s % 128, idx_s // 128)).T
        pointcloud[1,:3,:] = pc[idx_s,:].T
        # pc_n = imgs_normal[1][:,160:160*2].reshape(-1, 3)
        pc_n = normal[1].transpose(1,2,0)[:,160:160*2, :].reshape(-1, 3)[mask]
        pointcloud[1,3:6,:] = pc_n[idx_s,:].T
        # pc_c = imgs_rgb[1,:,160:160*2,:].reshape(-1,3)
        pc_c = imgs_rgb[1,:,160:160*2,:].reshape(-1,3)[mask]
        pointcloud[1,6:9,:] = pc_c[idx_s,::-1].T
        
        rets['pointcloud']=pointcloud[None,...]

    

    if self.local:
        
        R_s2t = np.matmul(R[1], R_inv[0])
        pointcloud[0,:3,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,:3,:]) + R_s2t[:3,3:4]
        pointcloud[0,3:6,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,3:6,:])
        
        #util.write_ply('test.ply', np.concatenate((pointcloud[0,:3,:].T,pointcloud[1,:3,:].T)),
        #  normal=np.concatenate((pointcloud[0,3:6,:].T,pointcloud[1,3:6,:].T)))
        if 1:
          N_PAIR_PTS = 6000
          N_PAIR_EXCEED_PTS = N_PAIR_PTS*10
          ANGLE_THRESH = 5.0
          PERP_THRESH = np.cos(np.deg2rad(90-ANGLE_THRESH))
          PARALLEL_THRESH = np.cos(np.deg2rad(ANGLE_THRESH))
          COPLANE_THRESH = 0.05
          rel_cls_pts = np.zeros([N_PAIR_EXCEED_PTS])
          ind_s = np.random.choice(pointcloud.shape[-1], N_PAIR_EXCEED_PTS)
          ind_t = np.random.choice(pointcloud.shape[-1], N_PAIR_EXCEED_PTS)
          pair_pts = np.stack((ind_s, ind_t), -1)
          normdot = (pointcloud[0, 3:6, pair_pts[:,0]] * pointcloud[1, 3:6, pair_pts[:,1]]).sum(1)
          dst = (np.abs(((pointcloud[0, 0:3, pair_pts[:,0]] - pointcloud[1, 0:3, pair_pts[:,1]]) * pointcloud[1, 3:6, pair_pts[:,1]]).sum(1)) + 
              np.abs(((pointcloud[0, 0:3, pair_pts[:,0]] - pointcloud[1, 0:3, pair_pts[:,1]]) * pointcloud[0, 3:6, pair_pts[:,0]]).sum(1)))/2
          rel_cls_pts[(np.abs(normdot) < PERP_THRESH)] = 1
          rel_cls_pts[(np.abs(normdot) > PARALLEL_THRESH) & (dst > COPLANE_THRESH)] = 2
          rel_cls_pts[(np.abs(normdot) > PARALLEL_THRESH) & (dst <= COPLANE_THRESH)] = 3

        if self.split == 'train':
          # balance each class
          N_CLASS = 4
          pair_pts_select=[]
          for j in range(N_CLASS):
            ind = np.where(rel_cls_pts == j)[0]
            if len(ind):
              pair_pts_select.append(ind[np.random.choice(len(ind), N_PAIR_PTS//N_CLASS)])
          pair_pts_select = np.concatenate(pair_pts_select)
          
          pair_pts_select =pair_pts_select[np.random.choice(len(pair_pts_select), N_PAIR_PTS)]
          pair_pts = pair_pts[pair_pts_select]
          normdot = normdot[pair_pts_select]
          dst = dst[pair_pts_select]
          rel_cls_pts = rel_cls_pts[pair_pts_select]
        else:
          pair_pts_select = np.random.choice(len(pair_pts), N_PAIR_PTS)
          pair_pts = pair_pts[pair_pts_select]
          normdot = normdot[pair_pts_select]
          dst = dst[pair_pts_select]
          rel_cls_pts = rel_cls_pts[pair_pts_select]


        rets['normdot2'] = np.power(normdot,2)[None,:]
        rets['dst2'] = np.power(dst,2)[None,:]
        # convert to image coordinate 
        
        if 1:
          R_t2s = np.linalg.inv(R_s2t)
          tp = (np.matmul(R_t2s[:3,:3], pointcloud[0, :3, pair_pts[:,0]].T)+R_t2s[:3,3:4]).T
          hfov = 90.0
          vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi))/np.pi*180
          
          zs = -tp[:,2]
          ys = (0.5 - (tp[:, 1]/zs/(np.tan(np.deg2rad(vfov/2))))/2)*160 
          xs = (0.5 + (tp[:, 0]/zs/(np.tan(np.deg2rad(hfov/2))))/2)*160
          uv_s = np.stack((xs, ys), -1)
          tp = pointcloud[1, :3, pair_pts[:,1]]
          zs = -tp[:,2]
          ys = (0.5 - (tp[:, 1]/zs/(np.tan(np.deg2rad(vfov/2))))/2)*160 
          xs = (0.5 + (tp[:, 0]/zs/(np.tan(np.deg2rad(hfov/2))))/2)*160
          uv_t = np.stack((xs, ys), -1)
          rets['uv_pts'] = np.stack((uv_s, uv_t))[None, :]
          rets['uv_pts'][:, :, :, 0] = rets['uv_pts'][:, :, :, 0].clip(0, 160-1)
          rets['uv_pts'][:, :, :, 1] = rets['uv_pts'][:, :, :, 1].clip(0, 160-1)
          rets['uv_pts'] = rets['uv_pts'].astype('int')

        rets['rel_cls_pts'] = rel_cls_pts[None, :]
        rets['pair_pts'] = pair_pts[None, :]


    if self.eval_local:
        
        # convert back into local coordinate 
        R_t2s = np.matmul(R[0], R_inv[1])
        Kth = self.dataList[index % self.__len__()]['Kth']
        pointcloud[0,:3,:] = np.matmul(R_t2s[:3,:3], pointcloud[0,:3,:]) + R_t2s[:3,3:4]
        pointcloud[0,3:6,:] = np.matmul(R_t2s[:3,:3], pointcloud[0,3:6,:])
        
        R_pred = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['pred_pose']
        gt_pose = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['gt_pose']

        err_r = util.angular_distance_np(R_pred[:3,:3],gt_pose[:3,:3])[0]
        rets['err_r'] = err_r
        
        rets['eval_key'] = '%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)
        pos_s_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['pos_s_360']
        pos_t_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['pos_t_360']
        nor_s_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['nor_s_360']
        nor_t_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['nor_t_360']
        feat_s_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['feat_s_360']
        feat_t_360 = self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, ct0, ct1, Kth)]['feat_t_360']
        if 1:
          rets['pos_s_360'] = (pos_s_360[None,:])
          rets['pos_t_360'] = (pos_t_360[None,:])
          rets['nor_s_360'] = (nor_s_360[None,:])
          rets['nor_t_360'] = (nor_t_360[None,:])

        #pointcloud[0,:3,:] = np.matmul(R_gt[:3,:3], pointcloud[0,:3,:]) + R_gt[:3,3:4]
        #pointcloud[0,3:6,:] = np.matmul(R_gt[:3,:3], pointcloud[0,3:6,:])
        #util.write_ply('test.ply', np.concatenate((pointcloud[0,:3,:].T,pointcloud[1,:3,:].T)),
        #  normal=np.concatenate((pointcloud[0,3:6,:].T,pointcloud[1,3:6,:].T)))
        pointcloud[0,:3,:] = np.matmul(R_pred[:3,:3], pointcloud[0,:3,:]) + R_pred[:3,3:4]
        pointcloud[0,3:6,:] = np.matmul(R_pred[:3,:3], pointcloud[0,3:6,:])
        igt = np.matmul(R_s2t, np.linalg.inv(R_pred))
        rets['igt'] = igt[None,:]
        rets['pred_pose'] = R_pred[None,:]
        rets['gt_pose'] = gt_pose[None,:]
        R_gt = igt[:3,:3]
        t_gt = igt[:3,3:4]
          

    else:
      delta_R = util.randomRotation(epsilon=0.1)
      delta_t = np.random.randn(3)*0.1
      
      pointcloud_s_perturb = np.matmul(delta_R, pointcloud[0,:3,:] - pointcloud[0,:3,:].mean(1)[:,None]) + delta_t[:, None] + pointcloud[0,:3,:].mean(1)[:,None]
      tp_R = delta_R 
      tp_t = np.matmul(np.eye(3) - delta_R, pointcloud[0,:3,:].mean(1)[:,None]) + delta_t[:, None]

      t_gt = np.matmul(np.eye(3) - delta_R.T, pointcloud[0,:3,:].mean(1)[:,None]) - np.matmul(delta_R.T, delta_t[:, None])
      R_gt = delta_R.T
      igt = np.eye(4)
      igt[:3,:3] = R_gt
      igt[:3,3] = t_gt.squeeze()
      rets['igt'] = igt[None,:]
      pointcloud_s_n_perturb = np.matmul(delta_R, pointcloud[0,3:6,:])
      pointcloud[0,:3,:] = pointcloud_s_perturb
      pointcloud[0,3:6,:] = pointcloud_s_n_perturb
    
    if 0:
      plane_params1[:,:4] = np.matmul(plane_params1[:,:4], igt)
    Q = np.concatenate((util.rot2Quaternion(R_gt),t_gt.squeeze()))
    R_ = np.eye(4)
    R_[:3, :3] = R_gt
    R_[:3, 3] = t_gt.squeeze()
    R_inv = np.linalg.inv(R_)
    rets['pointcloud']=pointcloud[None,...]

    if self.topdown:
        img2ind = np.zeros([2, self.num_points, 3])
        imgPCid = np.zeros([2,  self.num_points, 2])
        plane_eqs = np.zeros([2, 4])
        
        
        colors = np.random.rand(21,3)
        # resolution = 0.02 # 0.2m
        resolution = 0.03
        # height = 400 
        # width = 400
        height = 224
        width = 224
        # pc0 = # self.depth2pc(imgs[0][:,:160]).T
        pc0 = pointcloud[0,0:3,:].T
        pc2ind = np.zeros([2, len(pc0), 3])
        
        npts = np.zeros([2])
        pc2ind_mask = np.zeros([2, pointcloud.shape[2]])
        # pc = np.concatenate((pc, pc0))
        # the floor plane
        # (0, 1, 0)'x + d = 0
        
        # remove partial view's ceiling 
        dst = np.zeros([pc0.shape[0]])
        mask = dst < 1.5 
        # reorder pointcloud[0]
        
        validind = np.where(mask)[0]
        invalidind = np.where(~mask)[0]
        #pointcloud[0] = np.concatenate((pointcloud[0,:,validind].T,pointcloud[0,:,invalidind].T), -1)
        npts[0] = len(validind)
        pc0 = pc0[mask]
        pc2ind_mask[0] = mask

        # project camera position(0,0,0) to floor plane 
        plane_eq_0 = np.array([1,0.0,0.0,0.0])
        plane_eq = np.array([1,0.0,0.0,0.0])
        origin_0 = np.array([0.0,0.0,0.0])
        # axis [0,0,-1], []
        axis_base = np.array([0,0,-1])
        axis_y_0 = axis_base - np.dot(axis_base,plane_eq_0[:3]) * plane_eq_0[:3]
        axis_y_0 /= (np.linalg.norm(axis_y_0)+1e-16)
        axis_x_0 = np.cross(axis_y_0, plane_eq_0[:3])
        axis_x_0 /= (np.linalg.norm(axis_x_0)+1e-16)
        axis_z_0 = plane_eq_0[:3]

        
        topdown_c_partial_0 = np.zeros([224,224,3])
        topdown_c_partial_1 = np.zeros([224,224,3])
        topdown_c_complete_0 = np.zeros([224,224,3])
        topdown_c_complete_1 = np.zeros([224,224,3])


        topdown_s_complete_0 = np.zeros([224,224])
        topdown_s_complete_1 = np.zeros([224,224])

        tp = ~topdown_c_partial_0.sum(2).astype('bool')
        edt_0 = ndimage.distance_transform_edt(tp, return_indices=False)
        edt_0 = np.maximum(0.1, np.power(0.98, edt_0))
        tp = ~topdown_c_partial_1.sum(2).astype('bool')
        edt_1 = ndimage.distance_transform_edt(tp, return_indices=False)
        edt_1 = np.maximum(0.1, np.power(0.98, edt_1))
        rets['edt_w'] = np.stack((edt_0, edt_1))[None, ...]
        
        
        u = ((pc0 - origin_0[None,:]) * axis_x_0[None,:]).sum(1)
        v = ((pc0 - origin_0[None,:]) * axis_y_0[None,:]).sum(1)
        z = ((pc0 - origin_0[None,:]) * axis_z_0[None,:]).sum(1)
        # write_ply('test.ply',np.stack((u,v,z),-1), color=colors[pc_s])

        u = width//2 + (u / resolution).astype('int')
        v = height//2 - (v / resolution).astype('int')
        ind_z = np.digitize(z, [-0.1, 0.7, 1.5])
        topdown_ind_0 = np.stack((u, v, ind_z), -1)


        u = ((pointcloud[0,0:3,:].T - origin_0[None,:]) * axis_x_0[None,:]).sum(1)
        v = ((pointcloud[0,0:3,:].T - origin_0[None,:]) * axis_y_0[None,:]).sum(1)
        z = ((pointcloud[0,0:3,:].T - origin_0[None,:]) * axis_z_0[None,:]).sum(1)
        u = width//2 + (u / resolution).astype('int')
        v = height//2 - (v / resolution).astype('int')
        ind_z = np.digitize(z, [-0.1, 0.7, 1.5])
        topdown_ind_img_0 = np.stack((u, v, ind_z), -1)

        pc2ind[0,mask] = topdown_ind_0
        pc1 = pointcloud[1,0:3,:].T
        plane_eq_1 = np.matmul(plane_eq, np.linalg.inv(R[1]))
        plane_eq_1 /= (np.linalg.norm(plane_eq_1[:3])+1e-16)
        plane_eqs[1, :] = plane_eq_1.copy()
        
        dst = np.zeros([pc1.shape[0]])
        mask = dst < 1.5 
        
        validind = np.where(mask)[0]
        invalidind = np.where(~mask)[0]
        #pointcloud[1] = np.concatenate((pointcloud[1,:,validind].T,pointcloud[1,:,invalidind].T), -1)
        npts[1] = len(validind)
        pc1 = pc1[mask]
        pc2ind_mask[1] = mask
        
        origin_1 = -plane_eq_1[:3] * plane_eq_1[3]
        # axis [0,0,-1], []
        axis_base = np.array([0,0,-1])
        axis_y_1 = axis_base - np.dot(axis_base,plane_eq_1[:3]) * plane_eq_1[:3]
        axis_y_1 /= (np.linalg.norm(axis_y_1)+1e-16)
        axis_x_1 = np.cross(axis_y_1, plane_eq_1[:3])
        axis_x_1 /= (np.linalg.norm(axis_x_1)+1e-16)
        axis_z_1 = plane_eq_1[:3]

        u = ((pc1 - origin_1[None,:]) * axis_x_1[None,:]).sum(1)
        v = ((pc1 - origin_1[None,:]) * axis_y_1[None,:]).sum(1)
        z = ((pc1 - origin_1[None,:]) * axis_z_1[None,:]).sum(1)
        # write_ply('test.ply',np.stack((u,v,z),-1), color=colors[pc_s])

        u = width//2 + (u / resolution).astype('int')
        v = height//2 - (v / resolution).astype('int')
        ind_z = np.digitize(z, [-0.1, 0.7, 1.5])
        topdown_ind_1 = np.stack((u, v, ind_z), -1)

        u = ((pointcloud[1,0:3,:].T - origin_1[None,:]) * axis_x_1[None,:]).sum(1)
        v = ((pointcloud[1,0:3,:].T - origin_1[None,:]) * axis_y_1[None,:]).sum(1)
        z = ((pointcloud[1,0:3,:].T - origin_1[None,:]) * axis_z_1[None,:]).sum(1)
        u = width//2 + (u / resolution).astype('int')
        v = height//2 - (v / resolution).astype('int')
        ind_z = np.digitize(z, [-0.1, 0.7, 1.5])
        topdown_ind_img_1 = np.stack((u, v, ind_z), -1)
        

        img2ind[0] = topdown_ind_img_0
        img2ind[1] = topdown_ind_img_1
        pc2ind[1,mask] = topdown_ind_1
        rets['img2ind'] = img2ind[None,...]
        rets['imgPCid'] = imgPCid[None,...]
        rets['axis_x'] = np.zeros([2,3])
        rets['axis_y'] = np.zeros([2,3])
        rets['origin'] = np.zeros([2,3])
        

        rets['axis_x'][0] = axis_x_0
        rets['axis_y'][0] = axis_y_0
        rets['axis_x'][1] = axis_x_1
        rets['axis_y'][1] = axis_y_1
        rets['origin'][0] = origin_0
        rets['origin'][1] = origin_1
        rets['axis_x'] = rets['axis_x'][None,:]
        rets['axis_y'] = rets['axis_y'][None,:]
        rets['origin'] = rets['origin'][None,:]
        # sample points on source floor plane:
        if 1:
          #mask = ~((topdown_c_complete_0==0).sum(2)==3)
          
          mask = ~((topdown_c_partial_0==0).sum(2)==3)
          vs, us = np.where(mask)
          if not len(vs):
              vs = np.array([0,0])
              us = np.array([0,0])
          ind = np.random.choice(len(vs), 100)
          u_0 = us[ind]
          v_0 = vs[ind]
          
          #u_0 = np.random.choice(width, 100)
          #v_0 = np.random.choice(height, 100)

          kp_uv_0 = np.stack((u_0,v_0),-1)
          u_0 -= width//2
          v_0 -= height//2
         

          kp_3d_0 = origin_0[None,:] + axis_x_0[None,:] * u_0[:,None] * resolution - axis_y_0[None,:] * v_0[:,None] * resolution

          R01 = np.matmul(R[1], R_inv[0])
          kp_3d_1 = (np.matmul(R01[:3,:3], kp_3d_0.T) + R01[:3,3:4]).T

          # random sample a set of points as negative correspondencs 
          if 1:
            mask = ~((topdown_c_partial_1==0).sum(2)==3)
            vs_neg, us_neg = np.where(mask)
            if not len(vs_neg):
                vs_neg = np.array([0,0])
                us_neg = np.array([0,0])
            ind = np.random.choice(len(vs_neg), 100*100)
            u_neg_1 = us_neg[ind]
            v_neg_1 = vs_neg[ind]
            
            kp_uv_neg_1 = np.stack((u_neg_1,v_neg_1),-1)
            u_neg_1 -= width//2
            v_neg_1 -= height//2
            kp_3d_neg_1 = origin_1[None,:] + axis_x_1[None,:] * u_neg_1[:,None] * resolution - axis_y_1[None,:] * v_neg_1[:,None] * resolution
            R10 = np.matmul(R[0], R_inv[1])
            kp_3d_neg_0 = (np.matmul(R10[:3,:3], kp_3d_neg_1.T) + R10[:3,3:4]).T
            u_neg_0 = ((kp_3d_neg_0 - origin_0[None,:]) * axis_x_0[None,:]).sum(1)
            v_neg_0 = ((kp_3d_neg_0 - origin_0[None,:]) * axis_y_0[None,:]).sum(1)
            u_neg_0 = width//2 + (u_neg_0 / resolution).astype('int')
            v_neg_0 = height//2 - (v_neg_0 / resolution).astype('int')
            kp_uv_neg_0 = np.stack((u_neg_0,v_neg_0),-1)
            kp_uv_neg_0[:,0] = kp_uv_neg_0[:,0].clip(0, width-1)
            kp_uv_neg_0[:,1] = kp_uv_neg_0[:,1].clip(0, height-1)
            kp_uv_neg_1 = kp_uv_neg_1.reshape(100, 100, 2)
            kp_uv_neg_0 = kp_uv_neg_0.reshape(100, 100, 2)
          w_uv_neg_1 = 1 - np.maximum(0.1, np.power(0.98, np.linalg.norm(kp_uv_neg_0 - kp_uv_0[:, None, :], axis=2)))
          
          
          u_1 = ((kp_3d_1 - origin_1[None,:]) * axis_x_1[None,:]).sum(1)
          v_1 = ((kp_3d_1 - origin_1[None,:]) * axis_y_1[None,:]).sum(1)
          u_1 = width//2 + (u_1 / resolution).astype('int')
          v_1 = height//2 - (v_1 / resolution).astype('int')
          kp_uv_1 = np.stack((u_1,v_1),-1)
          

        topdown_c_complete = np.stack((topdown_c_complete_0, topdown_c_complete_1)).transpose(0,3,1,2)
        topdown_s_complete = np.stack((topdown_s_complete_0, topdown_s_complete_1))
        topdown_c_partial = np.stack((topdown_c_partial_0, topdown_c_partial_1))
 
        kp_uv_0[:,0] = kp_uv_0[:,0].clip(0, width-1)
        kp_uv_0[:,1] = kp_uv_0[:,1].clip(0, height-1)
        kp_uv_1[:,0] = kp_uv_1[:,0].clip(0, width-1)
        kp_uv_1[:,1] = kp_uv_1[:,1].clip(0, height-1)
        rets['kp_uv'] = np.stack((kp_uv_0,kp_uv_1))[None,...]
        rets['kp_uv_neg'] = kp_uv_neg_1[None,...]
        rets['w_uv_neg'] = w_uv_neg_1[None,...]
        rets['plane_eq'] = plane_eqs[None,...]
        rets['pc2ind'] = pc2ind[None,...]

        rets['pc2ind_mask'] = pc2ind_mask[None,...]
        rets['topdown'] = topdown_c_complete[None,...]
        rets['topdown_s'] = topdown_s_complete[None,...]
        rets['topdown_partial'] = topdown_c_partial.transpose(0,3,1,2)[None,...]
        
        TopDownValidMask = ((topdown_c_complete==0).sum(1,keepdims=True)!=3)
        rets['TopDownValidMask'] = TopDownValidMask[None,...]
        
        rets['npts'] = npts[None,...]

    if self.plane_m:
        if 0:
            scene_id = basePath.split('/')[-1]

            plane_file = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/Matterport_manual_plane/%s/' % self.split + scene_id + '.npy'

            plane_raw = np.load(plane_file,allow_pickle=True)

            plane_center = plane_raw[:,:3]
            plane_center = (np.matmul(R[0][:3,:3],plane_center.T)+R[0][:3,3:4]).T

            plane_normal = plane_raw[:,3:6]
            #plane_normal = (np.matmul(R[0][:3,:3],plane_normal.T)+R[0][:3,3:4]).T
            plane_normal = np.matmul(plane_normal, np.linalg.inv(R[0][:3,:3]))

            rets['plane_c'] = plane_center[np.newaxis,:]
            rets['plane_n'] = plane_normal[np.newaxis,:]
            rets['plane_raw'] = plane_raw[np.newaxis,:]
        
        MAX_PLANE = 20
        plane_ret = np.zeros([2, MAX_PLANE, 6])

        if self.split == 'train' or self.split == 'test':
            #import pdb; pdb.set_trace()
            scene_id = basePath.split(self.split)[-1]
            planePath = '/home/yzp12/projects/2020_CVPR_Hybrid/data/siming/Matterport3D_PCD/imagev2/'+self.split+scene_id
            plane_params = np.load(planePath + '/plane.npy',allow_pickle=True)
            plane_params = np.concatenate((np.zeros([1,9]),plane_params))
            imgs_plane_idx = np.zeros((self.nViews, 160,160, 3), dtype = np.float32)
            plane_idx_s = self.LoadImage(os.path.join(planePath,'plane_idx','%06d.png'%(ct0)),depth=False)[:,:,0]
            plane_idx_t = self.LoadImage(os.path.join(planePath,'plane_idx','%06d.png'%(ct1)),depth=False)[:,:,0]
            try:
                # remove planes that have  too little supports
                plane_idx_s[plane_params[plane_idx_s,-2]<1000] = 0
                plane_idx_t[plane_params[plane_idx_t,-2]<1000] = 0
            except:
                import pdb; pdb.set_trace()
            plane_idx_s = plane_idx_s.flatten()[mask_s]
            plane_idx_t = plane_idx_t.flatten()[mask_t]
            #import pdb; pdb.set_trace()

            plane_set_s = np.array(sorted([x for x in set(plane_idx_s)]))
            if len(plane_set_s) > MAX_PLANE:
                plane_set_s = plane_set_s[:MAX_PLANE]
            plane_params_s = plane_params[plane_set_s].copy()
            plane_ret[0, :len(plane_params_s), :3] = (np.matmul(R[0][:3,:3],plane_params_s[:,:3].T)+R[0][:3,3:4]).T
            plane_ret[0, :len(plane_params_s), 3:6] = np.matmul(plane_params_s[:,3:6], np.linalg.inv(R[0][:3,:3]))

            plane_set_t = np.array(sorted([x for x in set(plane_idx_t)]))
            if len(plane_set_t) > MAX_PLANE:
                plane_set_t = plane_set_t[:MAX_PLANE]
            plane_params_t = plane_params[plane_set_t].copy()
            plane_ret[1, :len(plane_params_t), :3] = (np.matmul(R[1][:3,:3],plane_params_t[:,:3].T)+R[1][:3,3:4]).T
            plane_ret[1, :len(plane_params_t), 3:6] = np.matmul(plane_params_t[:,3:6], np.linalg.inv(R[1][:3,:3]))

        
            n1 = len(plane_set_s)
            n2 = len(plane_set_t)
        
            planeID2newID_s = {plane_set_s[i]:i for i in range(n1)}
            planeID2newID_t = {plane_set_t[i]:i for i in range(n2)}

            plane_idx_s_new = np.zeros_like(plane_idx_s)
            for key in planeID2newID_s:
                plane_idx_s_new[plane_idx_s == key] = planeID2newID_s[key]
            plane_idx_t_new = np.zeros_like(plane_idx_t)
            for key in planeID2newID_t:
                plane_idx_t_new[plane_idx_t == key] = planeID2newID_t[key]

            plane_idx_new = np.stack((plane_idx_s_new, plane_idx_t_new))
        else:
            import ipdb;ipdb.set_trace()
            plane_params1, plane_idx1 = plane_utils.fit_planes(pointcloud[0,:3,:].T)
            plane_params2, plane_idx2 = plane_utils.fit_planes(pointcloud[0,:3,:].T)
            util.write_ply('test.ply', pointcloud[0,:3,:].T, color=np.random.rand(30,3)[plane_idx1.astype('int')])

            # plane_params[plane_set_s][:,3:6] = np.matmul(plane_params[plane_set_s][:,3:6], np.linalg.inv(R[0][:3,:3]))
            # util.write_ply('test.ply',pointcloud[0,:3,:].T, normal=np.matmul(plane_params[plane_idx_s,3:6], np.linalg.inv(R[0][:3,:3])))
            # util.write_ply('test.ply',pointcloud[0,:3,:].T, normal=plane_ret[0, :len(plane_params_s), 3:6][plane_idx_s_new])

        corres = []
        non_corres = []
      
        for key in plane_set_s:
            if key == 0: continue
            all_possible_t = [x for x in range(n2)]
            if key in plane_set_t:
                corres.append([planeID2newID_s[key], planeID2newID_t[key]])
                all_possible_t.remove(planeID2newID_t[key])
            for x in all_possible_t:
                non_corres.append([planeID2newID_s[key], x])
        if len(corres):
            corres = np.stack(corres)
        else:
            corres = np.zeros([0, 2])
        if len(non_corres):
            non_corres = np.stack(non_corres)
        else:
            non_corres = np.zeros([0, 2])

        MAX_PLANE_PAIR = 100
        plane_corres = np.zeros([MAX_PLANE_PAIR, 2])
        plane_noncorres = np.zeros([MAX_PLANE_PAIR, 2])
        if len(corres) < MAX_PLANE_PAIR:
            plane_corres[:len(corres),:] = corres
        else:
            plane_corres = corres[:MAX_PLANE_PAIR,:]
        if len(non_corres) < MAX_PLANE_PAIR:
            plane_noncorres[:len(non_corres),:] = non_corres
        else:
            plane_noncorres = non_corres[:MAX_PLANE_PAIR,:]
      

        plane_pointwise_c = np.zeros([plane_idx_new.shape[0], plane_idx_new.shape[1], 3])
        plane_pointwise_n = np.zeros([plane_idx_new.shape[0], plane_idx_new.shape[1], 3])
        plane_pointwise_mask = np.ones_like(plane_idx_new)
        plane_pointwise_mask[plane_idx_new==0] = 0
        plane_ret[:,0] = 0
        for k in range(2):
            plane_pointwise_c[k] = plane_ret[k,plane_idx_new[k],:3]
            plane_pointwise_n[k] = plane_ret[k,plane_idx_new[k],3:6]


        mask_plane = np.zeros([2, MAX_PLANE])
        mask_plane[0, :n1] = 1
        mask_plane[1, :n2] = 1
        rets['mask_plane'] = mask_plane[None, :]
        rets['plane_p'] = plane_ret[None, :]
        rets['num_plane_corres'] = np.array([len(corres)])[None, :]
        rets['num_plane_noncorres'] = np.array([len(non_corres)])[None, :]
        rets['plane_corres'] = plane_corres[None, :].astype('int')
        rets['num_plane'] = np.array([n1, n2])[None, :].astype('int')
        rets['plane_noncorres'] = plane_noncorres[None, :].astype('int')
        rets['plane_idx'] = plane_idx_new[None, :].astype('int')
        rets['plane_pointwise_c'] = plane_pointwise_c[None,:]
        rets['plane_pointwise_n'] = plane_pointwise_n[None,:]
        rets['plane_pointwise_mask'] = plane_pointwise_mask[None,:]

    if self.reproj:
      h=imgs.shape[1]
      pct,mask = util.depth2pc(imgs[1,:,160:160*2],'matterport')# be aware of the order of returned pc!!!
      ii=1
      colorpct=imgs_rgb[1,:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask,:]
      normalpct=normal_[0,1,:,:,ii*h:(ii+1)*h].reshape(3,-1).T[mask,:]
      depthpct=imgs[1,:,ii*h:(ii+1)*h].reshape(-1)[mask]
      # get the coordinates of each point in the first coordinate system
      
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

      pct,mask = util.depth2pc(imgs[0,:,160:160*2],'matterport')# be aware of the order of returned pc!!!
      colorpct=imgs_rgb[0,:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask]
      normalpct=normal_[0,0,:,:,ii*h:(ii+1)*h].reshape(3,-1).T[mask]
      depthpct=imgs[0,:,ii*h:(ii+1)*h].reshape(-1)[mask]
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

    for v in range(self.nViews):
      imgs_[v] =  cv2.resize(imgs[v], self.OutputSize,interpolation=cv2.INTER_NEAREST)
      if self.rgbd:
        imgs_rgb_[v] =  cv2.resize(imgs_rgb[v], self.OutputSize).transpose(2,0,1)
    
    imgs_ = imgs_[np.newaxis,:]
    if self.rgbd:
      imgs_rgb_ = imgs_rgb_[np.newaxis,:]
    R = R[np.newaxis,:]
    Q = Q[np.newaxis,:]
 

    rets['semantic']=segm
    rets['interval']=self.interval_this
    rets['norm']=normal_
    rets['rgb']=imgs_rgb_
    rets['depth']=imgs_
    rets['Q']=Q
    rets['R']=R
    rets['imgsPath']=imgsPath
    return rets
    
  def __len__(self):
    return self.len


