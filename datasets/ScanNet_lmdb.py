from open3d import *
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
import time
import random
import scipy.io as sio
import warnings
import lmdb
from util import rnd_color, perturb
from scipy import ndimage
from scipy.sparse import csc_matrix
from sklearn.neighbors import KDTree
import plane_utils


class ScanNet_lmdb(data.Dataset):
  def __init__(self, split, nViews, AuthenticdepthMap=False, crop=False, cache=True,\
        hmap=False,CorresCoords=False,meta=False,rotate=False,rgbd=False,birdview=False,pointcloud=False,num_points=8192,
        classifier=False,segm=False,segm_pyramid=False,normal=False,normal_pyramid=False,walls=False,gridPC=False,edges=False,samplePattern='',
        list_=None,singleView=True,siftFeatCorres=False,debug=False,orbfeat=False,siftPoint=False,denseCorres=False,reproj=False
        ,representation='skybox',entrySplit=None,dynamicWeighting=False,snumclass=0,corner=False,plane=True, plane_r=False, plane_m=False, small_debug=0, scannet_new_name=0, twoview_pointcloud=False,objectCloud=False,
        topdown=False,twoviewpointcloud=False,filter_overlap=None,local=False,fullsize_rgbdn=False,eval_local=False,local_method='point',local_eval_list=0):
    self.crop = crop
    self.local_method = local_method
    self.pointcloud = pointcloud
    self.twoviewpointcloud = twoviewpointcloud
    self.birdview = birdview
    self.num_points = num_points
    self.rgbd = rgbd
    self.local_eval_list = local_eval_list
    self.rotate = rotate
    self.meta = meta
    self.walls = walls
    self.eval_local = eval_local
    self.AuthenticdepthMap = AuthenticdepthMap
    self.hmap = hmap
    self.segm = segm
    self.plane = plane
    self.local = local
    self.twoview_pointcloud = twoview_pointcloud
    self.segm_pyramid = segm_pyramid
    self.representation = representation
    self.normal = normal
    self.fullsize_rgbdn=fullsize_rgbdn
    self.normal_pyramid = normal_pyramid
    self.samplePattern=samplePattern
    self.gridPC = gridPC
    self.edges = edges
    self.classifier = classifier
    self.CorresCoords = CorresCoords
    self.split = split
    self.nViews = nViews
    self.topdown = topdown
    self.singleView = singleView
    self.debug = debug
    self.siftFeatCorres = siftFeatCorres
    self.orbfeat = orbfeat
    self.siftPoint=siftPoint
    self.denseCorres=denseCorres
    self.objectCloud = objectCloud
    self.reproj=reproj
    self.corner = corner
    self.plane_r = plane_r
    self.plane_m = plane_m
    self.small_debug = small_debug
    self.entrySplit=entrySplit
    self.dynamicWeighting = dynamicWeighting
    if self.dynamicWeighting:
      assert(self.segm == True)
    self.snumclass = snumclass
    self.list = list_

    self.Inputwidth = 400 
    self.Inputheight = 200 
    self.nPanoView = 4
    self.cut = 224
    self.filter_overlap = filter_overlap
    self.intrinsic = np.array([[571.623718/640,0,319.500000/640],[0,571.623718/480,239.500000/480],[0,0,1]])
    self.intrinsicUnNorm = np.array([[571.623718,0,319.500000],[0,571.623718,239.500000],[0,0,1]])
    if self.plane_m:
        split = 'test'
        self.split = 'test'
    
    if 'scannet_test_scenes' in self.list:
        self.dataList = np.load(self.list, allow_pickle=True)
    else:
        self.dataList = np.load(self.list, allow_pickle=True).item()[split]#[:1]

    if os.path.exists('/scratch'):
      self.base = '/scratch/cluster/yzp12/projects/2020_CVPR_Hybrid/data/SkyBox/ScanNet/%s' % split
    else:
      self.base = '/home/yzp12/projects/2020_CVPR_Hybrid/data/SkyBox/ScanNet/%s' % split

    #split='test'
    if os.path.exists('/scratch'):
        lmdb_root = '../../../data/SkyBox/ScanNet120fov.%s.lmdb' % split
    else:
        lmdb_root = '/home/yzp12/mnt/eldar/projects/2020_CVPR_Hybrid//data/SkyBox/ScanNet120fov.%s.lmdb' % split
    if split=='test':
        if os.path.exists('/scratch'):
            lmdb_root = '../../../data/SkyBox/ScanNet120fovV2.%s.lmdb' % split
        else:
            lmdb_root = '/home/yzp12/mnt/eldar/projects/2020_CVPR_Hybrid//data/SkyBox/ScanNet120fovV2.%s.lmdb' % split
    self.env = lmdb.open(lmdb_root,
                 max_readers=1,
                 readonly=True,
                 lock=False,
                 readahead=False,
                 meminit=False)
    
    self.txn = self.env.begin(write=False) 

    if self.eval_local:
      self.eval_gt_dict = {}
      new_list=[]
      
      list_local = np.load(self.local_eval_list, allow_pickle=True)
      for i in range(len(list_local)):
        room_id = list_local[i][0]
        id_src = int(list_local[i][1])
        id_tgt = int(list_local[i][2])
        
        gt_pose = list_local[i][4]
        if self.txn.get(('%s-%06d-R' % (room_id, id_src)).encode()) and \
          self.txn.get(('%s-%06d-R' % (room_id, id_tgt)).encode()):
          for j in range(5):
            
            pred_pose = list_local[i][3][j]
            # pred_pose = np.eye(4)
            new_list.append({'base':self.dataList[0]['base'].split('scene')[0] + list_local[i][0],
            'id_src':id_src,
            'id_tgt':id_tgt,
            'Kth':j,
            'overlap':0,
            })
            self.eval_gt_dict['%s-%06d-%06d-%d' % (room_id, id_src, id_tgt, j)] = {'pred_pose':pred_pose,
                                                                              'gt_pose':gt_pose,
                                                                              'pos_s_360':np.zeros([1,3]),
                                                                              'pos_t_360':np.zeros([1,3]),
                                                                              'nor_s_360':np.zeros([1,3]),
                                                                              'nor_t_360':np.zeros([1,3]),
                                                                              'feat_s_360':np.zeros([1,3]),
                                                                              'feat_t_360':np.zeros([1,3]),
                                                                              }

        self.dataList = new_list
      
    if self.small_debug == 1:
        self.dataList = self.dataList * 100
    self.len = len(self.dataList)


    if self.entrySplit is not None:
      self.dataList = [self.dataList[kk] for kk in range(self.entrySplit*100,(self.entrySplit+1)*100)]
    
    
    print("datalist len:", self.len)
    #import pdb; pdb.set_trace()

    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    self.Rs = Rs

    Rs = np.zeros([6,4,4])
    Rs[0] = np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]]).T
    Rs[1] = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]]).T
    Rs[2] = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]).T
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]).T
    Rs[4] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]]).T
    Rs[5] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]]).T
    self.Rs = Rs
    
    # self.sift = cv2.xfeatures2d.SIFT_create()
  def drawMatch(self,img0,img1,src,tgt):
    if len(img0.shape)==2:
      img0=np.expand_dims(img0,2)
    if len(img1.shape)==2:
      img1=np.expand_dims(img1,2)
    h,w = img0.shape[0],img0.shape[1]
    img = np.zeros([2*h,w,3])
    img[:h,:,:] = img0
    img[h:,:,:] = img1
    n = len(src)
    for i in range(n):
      cv2.circle(img, (int(src[i,0]), int(src[i,1])), 3,(255,0,0),-1)
      cv2.circle(img, (int(tgt[i,0]), int(tgt[i,1])+h), 3,(255,0,0),-1)
      cv2.line(img, (int(src[i,0]),int(src[i,1])),(int(tgt[i,0]),int(tgt[i,1])+h),(255,0,0),1)
    return img

  def depth2pc(self, depth,needmask=False):
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

    if needmask:
      return pc,mask
    else:
      return pc
  def LoadImage(self, PATH,depth=True):

    # print(PATH)

    if depth:
      img = cv2.imread(PATH,2)/1000.
    else:
      img = cv2.imread(PATH) # load in rgb format
    if img.shape[1] == 960:
      img = img[:, 160*2:]
    return img
  
  def shuffle(self):
    pass
  
  def __getpair__(self, index):
    index = index % len(self.dataList)
    self.base_this = self.dataList[index]['base']
    self.interval_this = '0-15'
    ct0,ct1=self.dataList[index]['id_src'],self.dataList[index]['id_tgt']

    return ct0,ct1
  def plot_plane(self, pointcloud, plane_idx, plane_params):
    v = [pointcloud[:,:3][plane_idx ==0]]
    # v_c = [pointcloud[:,6:9][plane_idx == 0]]
    v_c = [np.tile(np.random.rand(3)[None,:], [len(v[-1]),1])]
    for i in range(len(plane_params)):
      center = pointcloud[:,:3][plane_idx == i+1].mean(0)
      color = np.random.rand(3)
      v.append(pointcloud[:,:3][plane_idx == i+1])
      v_c.append(np.tile(color[None,:], [len(v[-1]),1]))
      if 0:
        vert_plane, color_plane = get_plane_verts(plane_params[i], center, color)
        if plane_params[i][4] > 600:
          v.append(vert_plane)
          v_c.append(color_plane)
    v = np.concatenate(v)
    v_c = np.concatenate(v_c)
    util.write_ply('test.ply', v, color=v_c)
  def __getitem__(self, index):
    while True:
      #st=time.time()
      try:
        for i in range(self.__len__()):
          ct0,ct1 = self.__getpair__(i)


        ret, valid = self.__getitem__helper(index)
      except Exception as e:
          import ipdb;ipdb.set_trace()
          print(e)
          valid = False
          #import ipdb; ipdb.set_trace() 
      #print('time for load one data: %.3f' % (time.time() - st))
      if valid:
        break 
      else:
        index = np.random.choice(self.__len__(), 1)[0]
    return ret
  
  def __getitem__helper(self, index):
      #import ipdb;ipdb.set_trace()
      rets = {}
      index = index % self.__len__()
      imgs_depth = np.zeros((self.nViews, self.Inputheight, self.Inputwidth), dtype = np.float32)
      imgs_s = np.zeros((self.nViews, self.Inputheight, self.Inputwidth), dtype = np.float32)
      imgs_rgb = np.zeros((self.nViews, self.Inputheight, self.Inputwidth,3), dtype = np.float32)
      imgs_normal = np.zeros((self.nViews, self.Inputheight, self.Inputwidth,3), dtype = np.float32)
      pointcloud = np.zeros((self.nViews, 3+3+3+1, self.num_points), dtype = np.float32)
      
      R = np.zeros((self.nViews, 4, 4))
      Q = np.zeros((7))
      assert(self.nViews == 2)
      imgsPath = []
      ct0,ct1 = self.__getpair__(index)
      
      if 'scannet_test_scenes' not in self.list:
        rets['overlap'] = float(self.dataList[index]['overlap'])
      
      room_id = self.base_this.split('/')[-1]

      basePath = os.path.join(self.base, room_id)

      
      imageKey = '%s-%06d-rgb' % (room_id, ct0)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_rgb[0] = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.0
      imageKey = '%s-%06d-rgb' % (room_id, ct1)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_rgb[1] = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.0

      
      imageKey = '%s-%06d-depth' % (room_id, ct0)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_depth[0] = cv2.imdecode(imageBuf, 2).astype('float')/1000.0
      imageKey = '%s-%06d-depth' % (room_id, ct1)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_depth[1] = cv2.imdecode(imageBuf, 2).astype('float')/1000.0
      

      #cv2.imwrite('test.png',imgs_rgb[0]*255)
      imageKey = '%s-%06d-normal' % (room_id, ct0)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_normal[0] = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.0*2-1
      imageKey = '%s-%06d-normal' % (room_id, ct1)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_normal[1] = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.0*2-1
      
      imageKey = '%s-%06d-semantic' % (room_id, ct0)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_s[0] = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')[:,:,0]
      imageKey = '%s-%06d-semantic' % (room_id, ct1)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_s[1] = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')[:,:,0]
      
      PerspectiveValidMask = (imgs_depth!=0)
      rets['PerspectiveValidMask'] = PerspectiveValidMask[None,:,None,:,:]
      rets['dataMask'] = rets['PerspectiveValidMask']
      
      RKey = '%s-%06d-R' % (room_id, ct0)
      R[0] = np.frombuffer(self.txn.get(RKey.encode()), np.float).reshape(4,4)
      
      RKey = '%s-%06d-R' % (room_id, ct1)
      R[1] = np.frombuffer(self.txn.get(RKey.encode()), np.float).reshape(4,4)
      # convert from 3rd view to 4th view
      
      #R[0] = np.matmul(np.linalg.inv(self.Rs[3]),R[0])
      #R[1] = np.matmul(np.linalg.inv(self.Rs[3]),R[1])
      
      R_inv = np.linalg.inv(R)
      img2ind = np.zeros([2, self.num_points, 3])
      imgPCid = np.zeros([2,  self.num_points, 2])
      
      if self.fullsize_rgbdn:
        imgs_rgb_full = np.zeros((self.nViews, 480,640, 3), dtype = np.float32)
        imgs_norm_full = np.zeros((self.nViews, 480,640, 3), dtype = np.float32)
        imgs_full = np.zeros((self.nViews, 480,640), dtype = np.float32)
        imgs_full[0] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_depth','%06d.png'%(ct0))).copy()
        imgs_full[1] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_depth','%06d.png'%(ct1))).copy()
        imgs_rgb_full[0] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_rgb','%06d.png'%(ct0)),depth=False).copy()/255.
        imgs_rgb_full[1] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_rgb','%06d.png'%(ct1)),depth=False).copy()/255.
        imgs_norm_full[0] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_normal','%06d.png'%(ct0)),depth=False).copy()/255*2-1.
        imgs_norm_full[1] = self.LoadImage(os.path.join(basePath.replace('ScanNet','ScanNet_360'),'obs_normal','%06d.png'%(ct1)),depth=False).copy()/255*2-1.
        rets['rgb_full'] = imgs_rgb_full[np.newaxis,:]
        rets['norm_full'] = imgs_norm_full[np.newaxis,:]
        rets['depth_full'] = imgs_full[np.newaxis,:]
      
      if self.denseCorres:
        
        # get 3d point cloud for each pano
        pcs,masks = self.depth2pc(imgs_depth[0],needmask=True) # be aware of the order of returned pc!!!
        pct,maskt = self.depth2pc(imgs_depth[1],needmask=True)
        pct = (np.matmul(R_inv[1][:3,:3], pct.T) + R_inv[1][:3,3:4]).T
        pcs = (np.matmul(R_inv[0][:3,:3], pcs.T) + R_inv[0][:3,3:4]).T
        inds = np.arange(imgs_depth[0].shape[0]*imgs_depth[0].shape[1])[masks]
        indt = np.arange(imgs_depth[0].shape[0]*imgs_depth[0].shape[1])[maskt]
        # find correspondence using kdtree
        tree = KDTree(pct)
        IdxQuery=np.random.choice(range(pcs.shape[0]),5000)
        # sample 5000 query points
        pcsQuery = pcs[IdxQuery,:]
        pcsQueryid = inds[IdxQuery]
        nearest_dist, nearest_ind = tree.query(pcsQuery, k=1)
        hasCorres=(nearest_dist < 0.08)
        idxTgtNeg=[]
        idxSrc= np.stack((pcsQueryid[hasCorres[:,0]] % self.Inputwidth, pcsQueryid[hasCorres[:,0]]// self.Inputwidth),1)
        idxTgt= np.stack((indt[nearest_ind[hasCorres]] % self.Inputwidth, indt[nearest_ind[hasCorres]] // self.Inputwidth),1)
        
        if hasCorres.sum() < 200:
          rets['denseCorres']={'idxSrc':np.zeros([1,500,2]).astype('int'),'idxTgt':np.zeros([1,500,2]).astype('int'),'valid':np.array([0]),'idxTgtNeg':idxTgtNeg}
        else:

          idx2000 = np.random.choice(range(idxSrc.shape[0]),500)
          idxSrc=idxSrc[idx2000][np.newaxis,:]
          idxTgt=idxTgt[idx2000][np.newaxis,:]
          rets['denseCorres']={'idxSrc':idxSrc.astype('int'),'idxTgt':idxTgt.astype('int'),'valid':np.array([1]),'idxTgtNeg':idxTgtNeg}

      
      if self.pointcloud or self.local:

        #pc = self.depth2pc(imgs_depth[0][:,160:160*2]).T
        pc, mask = self.depth2pc(imgs_depth[0][100-48:100+48,200-64:200+64], needmask=True)
        
        # util.write_ply('test.ply',np.concatenate((pc,pc1)))
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        mask_s = np.where(mask)[0][idx_s]
        
        imgPCid[0] = np.stack((idx_s % 128, idx_s // 128)).T
        pointcloud[0,:3,:] = pc[idx_s,:].T
        
        pc_n = imgs_normal[0][100-48:100+48,200-64:200+64].reshape(-1, 3)[mask]
        pointcloud[0,3:6,:] = pc_n[idx_s,:].T
        
        pc_c = imgs_rgb[0][100-48:100+48,200-64:200+64].reshape(-1,3)[mask]
        pointcloud[0,6:9,:] = pc_c[idx_s,::-1].T
        
        pc_s = imgs_s[0][100-48:100+48,200-64:200+64].reshape(-1)[mask]
        pointcloud[0,9:10,:] = pc_s[idx_s]

        
        pc, mask = self.depth2pc(imgs_depth[1][100-48:100+48,200-64:200+64], needmask=True)
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        mask_t = np.where(mask)[0][idx_s]
        
        imgPCid[1] = np.stack((idx_s % 128, idx_s // 128)).T
        pointcloud[1,:3,:] = pc[idx_s,:].T
        
        pc_n = imgs_normal[1][100-48:100+48,200-64:200+64].reshape(-1, 3)[mask]
        pointcloud[1,3:6,:] = pc_n[idx_s,:].T
        
        pc_c = imgs_rgb[1][100-48:100+48,200-64:200+64].reshape(-1,3)[mask]
        pointcloud[1,6:9,:] = pc_c[idx_s,::-1].T
        
        pc_s = imgs_s[1][100-48:100+48,200-64:200+64].reshape(-1)[mask]
        pointcloud[1,9:10,:] = pc_s[idx_s]
        rets['pointcloud']=pointcloud[None,...]
        
      if self.plane_r:
        Key = '%s-plane' % (room_id)
        plane_eq_raw = np.frombuffer(self.txn.get(Key.encode()), np.float).reshape(-1,9)
        Key = '%s-plane-validnum' % (room_id)
        valid_plane = np.frombuffer(self.txn.get(Key.encode()),np.uint8)[0]
        plane_eq = plane_eq_raw[:,3:7]
        plane_eq = np.matmul(plane_eq, np.linalg.inv(R[0]))
        plane_center = plane_eq_raw[:,:3]
        plane_center = (np.matmul(R[0][:3,:3], plane_center.T) + R[0][:3,3:4]).T
        
        rets['plane']=plane_eq[np.newaxis,:]
        rets['plane_raw']=plane_eq_raw[np.newaxis,:]
        rets['plane_c']=plane_center[np.newaxis,:]
        rets['valid_plane']=valid_plane
      
      
      

      if self.local:
        # sample point-level relation from plane relation
        
        try:        
          R_s2t = np.matmul(R[1], R_inv[0])
          pointcloud[0,:3,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,:3,:]) + R_s2t[:3,3:4]
          pointcloud[0,3:6,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,3:6,:])
          if self.eval_local:
            N_PAIR_PTS = 6000
          else:
            N_PAIR_PTS = 1000
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
            R_s2t = np.matmul(R[1], R_inv[0])
            R_t2s = np.linalg.inv(R_s2t)
            tp = (np.matmul(R_t2s[:3,:3], pointcloud[0, :3, pair_pts[:,0]].T)+R_t2s[:3,3:4]).T
            hfov = 120.0
            vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*200/400)/np.pi*180

            zs = -tp[:,2]
            ys = (0.5 - (tp[:, 1]/96*200/zs/(np.tan(np.deg2rad(vfov/2))))/2)*96 
            xs = (0.5 + (tp[:, 0]/128*400/zs/(np.tan(np.deg2rad(hfov/2))))/2)*128
            uv_s = np.stack((xs, ys), -1)
            tp = pointcloud[1, :3, pair_pts[:,1]]
            zs = -tp[:,2]
            ys = (0.5 - (tp[:, 1]/96*200/zs/(np.tan(np.deg2rad(vfov/2))))/2)*96 
            xs = (0.5 + (tp[:, 0]/128*400/zs/(np.tan(np.deg2rad(hfov/2))))/2)*128
            uv_t = np.stack((xs, ys), -1)
            rets['uv_pts'] = np.stack((uv_s, uv_t))[None, :]
            rets['uv_pts'][:, :, :, 0] = rets['uv_pts'][:, :, :, 0].clip(0, 128-1)
            rets['uv_pts'][:, :, :, 1] = rets['uv_pts'][:, :, :, 1].clip(0, 96-1)
            rets['uv_pts'] = rets['uv_pts'].astype('int')
        except:
          import ipdb;ipdb.set_trace()
        
        
          rel_cls = np.array(rel_cls)
          rel_dst = np.array(rel_dst)
          rel_ndot = np.array(rel_ndot)
          pair = np.concatenate(pair).reshape(-1, 2)
          

            
          # padding f
          MAX_PAIR = 100
          MAX_PLANE = 20
          plane_params1 = np.array(plane_params1)
          plane_params2 = np.array(plane_params2)
          if len(plane_params1) <= MAX_PLANE:
            plane_params1 = np.concatenate((plane_params1, np.zeros([MAX_PLANE - len(plane_params1), 5])))
            plane_center1 = np.concatenate((plane_center1, np.zeros([MAX_PLANE - len(plane_center1), 6])))
          else:
            plane_params1 = plane_params1[:MAX_PLANE]
            plane_center1 = plane_center1[:MAX_PLANE]
            select = (pair[:, 0] < MAX_PLANE)
            pair = pair[select]
            rel_cls = rel_cls[select]
            rel_dst = rel_dst[select]
            rel_ndot = rel_ndot[select]
          if len(plane_params2) <= MAX_PLANE:
            plane_params2 = np.concatenate((plane_params2, np.zeros([MAX_PLANE - len(plane_params2), 5])))
            plane_center2 = np.concatenate((plane_center2, np.zeros([MAX_PLANE - len(plane_center2), 6])))
          else:
            plane_params2 = plane_params2[:MAX_PLANE]
            plane_center2 = plane_center2[:MAX_PLANE]
            select = (pair[:, 1] < MAX_PLANE)
            pair = pair[select]
            rel_cls = rel_cls[select]
            rel_dst = rel_dst[select]
            rel_ndot = rel_ndot[select]
          rel_valid = np.zeros([MAX_PAIR])
          if len(rel_cls) < MAX_PAIR:
            rel_valid[:len(rel_cls)] = 1
            rel_cls = np.concatenate((rel_cls, np.zeros([MAX_PAIR - len(rel_cls)])))
            rel_dst = np.concatenate((rel_dst, np.zeros([MAX_PAIR - len(rel_dst)])))
            rel_ndot = np.concatenate((rel_ndot, np.zeros([MAX_PAIR - len(rel_ndot)])))
            pair = np.concatenate((pair, np.zeros([MAX_PAIR - len(pair), 2])))
          else:
            pair = pair[:MAX_PAIR]
            rel_cls = rel_cls[:MAX_PAIR]
            rel_dst = rel_dst[:MAX_PAIR]
            rel_ndot = rel_ndot[:MAX_PAIR]
            rel_valid[:] = 1
          rets['plane_center'] = np.stack((plane_center1,plane_center2))[None,...]
          rets['pair'] = pair[None,...].astype('int')
          rets['rel_cls'] = rel_cls[None,...].astype('int')
          rets['rel_dst'] = rel_dst[None,...]
          rets['rel_ndot'] = rel_ndot[None,...]
          rets['rel_valid'] = rel_valid[None,...]
          rets['plane_idx'] = np.stack((plane_idx1,plane_idx2))[None,...].astype('int')
        
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
            
          # transform source
          pos_s_360 = (np.matmul(R_pred[:3,:3], pos_s_360.T) + R_pred[:3,3:4]).T
          nor_s_360 = np.matmul(R_pred[:3,:3], nor_s_360.T).T
          
          
          # find top correspondence 
          if 0:
            tree = KDTree(pos_s_360)
            nearest_dist1, nearest_ind1 = tree.query(pos_t_360, k=1)
            nearest_ind1 = nearest_ind1.squeeze()
            tree = KDTree(pos_t_360)
            nearest_dist2, nearest_ind2 = tree.query(pos_s_360, k=1)
            nearest_ind2 = nearest_ind2.squeeze()
            # if nearest_ind1[nearest_ind2] == np.range(len(feat_s_360))
            rets['pos_s_360'] = (pos_s_360[nearest_ind1][None,:])
            rets['pos_t_360'] = (pos_t_360[None,:])
            rets['nor_s_360'] = (nor_s_360[nearest_ind1][None,:])
            rets['nor_t_360'] = (nor_t_360[None,:])

          if 1:
            rets['pos_s_360'] = (pos_s_360[None,:])
            rets['pos_t_360'] = (pos_t_360[None,:])
            rets['nor_s_360'] = (nor_s_360[None,:])
            rets['nor_t_360'] = (nor_t_360[None,:])
          
          
          pointcloud[0,:3,:] = np.matmul(R_pred[:3,:3], pointcloud[0,:3,:]) + R_pred[:3,3:4]
          pointcloud[0,3:6,:] = np.matmul(R_pred[:3,:3], pointcloud[0,3:6,:])

          
          color_t_360 = np.tile(np.array([0,1,0])[None,:], [len(pos_t_360),1])

          igt = np.matmul(R_s2t, np.linalg.inv(R_pred))
          rets['igt'] = igt[None,:]
          rets['pred_pose'] = R_pred[None,:]
          rets['gt_pose'] = gt_pose[None,:]
          R_gt = igt[:3,:3]
          t_gt = igt[:3,3:4]
        else:
          delta_R = util.randomRotation(epsilon=0.1*3)
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
          # np.matmul(R_gt, pointcloud_s_perturb) + t_gt
          
          if self.local_method == 'patch':
            plane_params1[:,:4] = np.matmul(plane_params1[:,:4], igt)
          Q = np.concatenate((util.rot2Quaternion(R_gt),t_gt.squeeze()))
          R_ = np.eye(4)
          R_[:3, :3] = R_gt
          R_[:3, 3] = t_gt.squeeze()
          R_inv = np.linalg.inv(R_)
          
          pointcloud[0,:3,:] = pointcloud_s_perturb
          pointcloud[0,3:6,:] = pointcloud_s_n_perturb
        rets['pointcloud']=pointcloud[None,...]
        
        
      if self.topdown:
        
        Key = '%s-pc' % (room_id)
        roompc = np.frombuffer(self.txn.get(Key.encode()), np.float).reshape(-1,3)
        roompc = roompc[np.random.choice(roompc.shape[0],20000)]
        rets['roompc'] = roompc[None,:]

        Key = '%s-floor' % (room_id)
        plane_eq = np.frombuffer(self.txn.get(Key.encode()), np.float).reshape(4)
        plane_eqs = np.zeros([2, 4])
        plane_eq_0 = np.matmul(plane_eq, np.linalg.inv(R[0]))
        plane_eq_0 /= (np.linalg.norm(plane_eq_0[:3])+1e-16)
        plane_eqs[0, :] = plane_eq_0.copy()
        plane_eq_1 = np.matmul(plane_eq, np.linalg.inv(R[1]))
        plane_eq_1 /= (np.linalg.norm(plane_eq_1[:3])+1e-16)
        plane_eqs[1, :] = plane_eq_1.copy()

        colors = np.random.rand(21,3)

        resolution = 0.03

        height = 224
        width = 224

        pc0 = pointcloud[0,0:3,:].T
        pc2ind = np.zeros([2, len(pc0), 3])
        
        npts = np.zeros([2])
        pc2ind_mask = np.zeros([2, pointcloud.shape[2]])

        # the floor plane
        # (0, 1, 0)'x + d = 0
        
        # remove partial view's ceiling 
        dst = np.abs(((plane_eq_0[:3][None,:] * pc0).sum(1) + plane_eq_0[3]))
        mask = dst < 1.5 
        # reorder pointcloud[0]
        
        validind = np.where(mask)[0]
        invalidind = np.where(~mask)[0]
        #pointcloud[0] = np.concatenate((pointcloud[0,:,validind].T,pointcloud[0,:,invalidind].T), -1)
        npts[0] = len(validind)
        pc0 = pc0[mask]
        pc2ind_mask[0] = mask

        # project camera position(0,0,0) to floor plane 
        origin_0 = -plane_eq_0[:3] * plane_eq_0[3]
        # axis [0,0,-1], []
        axis_base = np.array([0,0,-1])
        axis_y_0 = axis_base - np.dot(axis_base,plane_eq_0[:3]) * plane_eq_0[:3]
        axis_y_0 /= (np.linalg.norm(axis_y_0)+1e-16)
        axis_x_0 = np.cross(axis_y_0, plane_eq_0[:3])
        axis_x_0 /= (np.linalg.norm(axis_x_0)+1e-16)
        axis_z_0 = plane_eq_0[:3]

        
        imageKey = '%s-%06d-topdown_c_partial' % (room_id, ct0)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_c_partial_0 = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.
        imageKey = '%s-%06d-topdown_c_partial' % (room_id, ct1)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_c_partial_1 = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.

        imageKey = '%s-%06d-topdown_c_complete' % (room_id, ct0)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_c_complete_0 = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.
        imageKey = '%s-%06d-topdown_c_complete' % (room_id, ct1)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_c_complete_1 = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR).astype('float')/255.

        
        imageKey = '%s-%06d-topdown_s_complete' % (room_id, ct0)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_s_complete_0 = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')
        imageKey = '%s-%06d-topdown_s_complete' % (room_id, ct1)
        imageBin = self.txn.get(imageKey.encode())
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        topdown_s_complete_1 = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')

        
        
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
        dst = np.abs(((plane_eq_1[:3][None,:] * pc1).sum(1) + plane_eq_1[3]))
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
          
          # visualize correspondence 
          if 0:
            img_vis = (np.concatenate((topdown_c_complete_0,topdown_c_complete_1))*255).astype('uint8')
            for j in range(10):
              ind = np.random.choice(len(kp_uv_0),1)[0]
              img_vis = cv2.line(img_vis, (kp_uv_0[ind][0], kp_uv_0[ind][1]), (kp_uv_1[ind][0], kp_uv_1[ind][1]+topdown_c_complete_0.shape[0]), (255,255,0))
            cv2.imwrite('test.png',img_vis)

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


      imgsPath.append(f"{basePath}/{ct0:06d}")
      imgsPath.append(f"{basePath}/{ct1:06d}")
      
      rets['norm']=imgs_normal.transpose(0,3,1,2)[None,...]
      rets['rgb']=imgs_rgb.transpose(0,3,1,2)[None,...]
      rets['semantic']=imgs_s[None,...]
      rets['depth']=imgs_depth[None,:,None,:,:]
      rets['Q']=Q[None,...]
      rets['R']=R[None,...]
      rets['R_inv'] = R_inv[None,...]
      rets['imgsPath']=imgsPath
      

      return rets, True

  def __len__(self):
    return self.len






