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
from scipy import ndimage
from scipy.sparse import csc_matrix
from sklearn.neighbors import KDTree
import plane_utils


class SUNCG_lmdb(data.Dataset):
  def __init__(self, split, nViews, AuthenticdepthMap=False, crop=False, cache=True,\
        hmap=False,CorresCoords=False,meta=False,rotate=False,rgbd=False,birdview=False,pointcloud=False,num_points=8192,
        classifier=False,segm=False,segm_pyramid=False,normal=False,normal_pyramid=False,walls=False,gridPC=False,edges=False,samplePattern='',
        list_=None,singleView=True,siftFeatCorres=False,debug=False,orbfeat=False,siftPoint=False,denseCorres=False,reproj=False
        ,representation='skybox',entrySplit=None,dynamicWeighting=False,snumclass=0,corner=False,plane=True, plane_r=False,plane_m=False,scannet_new_name=False,twoview_pointcloud=False,objectCloud=False,
        topdown=False,twoviewpointcloud=False,filter_overlap=None,local=None,eval_local=False,local_method='point',local_eval_list=None):
    self.crop = crop
    self.pointcloud = pointcloud
    self.twoviewpointcloud = twoviewpointcloud
    self.birdview = birdview
    self.num_points = num_points
    self.rgbd = rgbd
    self.rotate = rotate
    self.meta = meta
    self.local_method = local_method
    self.local_eval_list = local_eval_list
    self.walls = walls
    self.AuthenticdepthMap = AuthenticdepthMap
    self.hmap = hmap
    self.segm = segm
    self.plane = plane
    self.local = local
    self.eval_local = eval_local
    self.twoview_pointcloud = twoview_pointcloud
    self.segm_pyramid = segm_pyramid
    self.representation = representation
    self.normal = normal
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
    self.entrySplit=entrySplit
    self.dynamicWeighting = dynamicWeighting
    if self.dynamicWeighting:
      assert(self.segm == True)
    self.snumclass = snumclass
    self.list = list_
    self.OutputSize = (640,160)
    self.Inputwidth = config.pano_width
    self.Inputheight = config.pano_height
    self.nPanoView = 4
    self.cut = 224
    self.filter_overlap = filter_overlap
    self.intrinsic = np.array([[571.623718/640,0,319.500000/640],[0,571.623718/480,239.500000/480],[0,0,1]])
    self.intrinsicUnNorm = np.array([[571.623718,0,319.500000],[0,571.623718,239.500000],[0,0,1]])
    self.dataList = np.load(self.list, allow_pickle=True).item()[split]#[:1]

    lmdb_root = '../data/SkyBox/SUNCGFixNormal.%s.lmdb' % split

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
          for j in range(1):
            pred_pose = list_local[i][3]
            sceneid, roomid = list_local[i][0].split('-')
            new_list.append({'base':self.dataList[0]['base'].split('SUNCG')[0] +'SUNCG/test/' + sceneid + '/' + roomid,
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
      

    self.len = len(self.dataList)

    if self.entrySplit is not None:
      self.dataList = [self.dataList[kk] for kk in range(self.entrySplit*100,(self.entrySplit+1)*100)]
    
    
    print("datalist len:", self.len)

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
  def depth2pc(self, depth,needmask=False):
    w,h = depth.shape[1], depth.shape[0]
    ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
    ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
    zs = depth.flatten()
    if w == 640 and h == 480:
      ys, xs = ys.flatten()*zs/1.73205081, xs.flatten()*zs/1.29903811
    elif w == 160 and h == 256:
      ys, xs = ys.flatten()*zs/0.3249197, xs.flatten()*zs/0.51987151
    elif w == 160 and h == 200:
      ys, xs = ys.flatten()*zs/0.66817864, xs.flatten()*zs/0.8352233
    elif w == 160 and h == 160:
      #ys, xs = ys.flatten()*zs/1.732, xs.flatten()*zs/1.732
      ys, xs = ys.flatten()*zs, xs.flatten()*zs
    mask = (zs!=0)
    pts = np.concatenate((xs[mask],ys[mask],-zs[mask])).reshape(3,-1)
    if needmask:
      return pts,mask
    else:
      return pts
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

  def __getitem__(self, index):
    
    while True:
      #st=time.time()
      try:
        
        ret, valid = self.__getitem__helper(index)
      except Exception as e:
          print(e)
          valid = False
      #print('time for load one data: %.3f' % (time.time() - st))
      if valid:
        break 
      else:
        index = np.random.choice(self.__len__(), 1)[0]
    
    return ret
  
  def __getitem__helper(self, index):

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
      
      rets['overlap'] = float(self.dataList[index]['overlap'])
      
      basePath = self.base_this
      scene_id = basePath.split('/')[-2]
      room_id = scene_id + '-' + basePath.split('/')[-1]
      
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
      imgs_s[0] = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')[:,:,0]+1
      imageKey = '%s-%06d-semantic' % (room_id, ct1)
      imageBin = self.txn.get(imageKey.encode())
      imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
      imgs_s[1] = cv2.imdecode(imageBuf, cv2.IMREAD_UNCHANGED).astype('uint8')[:,:,0]+1
      
      PerspectiveValidMask = (imgs_depth!=0)
      rets['PerspectiveValidMask'] = PerspectiveValidMask[None,:,None,:,:]
      rets['dataMask'] = rets['PerspectiveValidMask']

      
      RKey = '%s-%06d-R' % (room_id, ct0)
      R[0] = np.frombuffer(self.txn.get(RKey.encode()), np.float).reshape(4,4)
      
      RKey = '%s-%06d-R' % (room_id, ct1)
      R[1] = np.frombuffer(self.txn.get(RKey.encode()), np.float).reshape(4,4)
      # convert from 3rd view to 4th view
      
      R[0] = np.matmul(np.linalg.inv(self.Rs[3]),R[0])
      R[1] = np.matmul(np.linalg.inv(self.Rs[3]),R[1])
      
      R_inv = np.linalg.inv(R)
      img2ind = np.zeros([2, self.num_points, 3])
      imgPCid = np.zeros([2,  self.num_points, 2])
      
      

      if self.pointcloud or self.local:
        pc = self.depth2pc(imgs_depth[0][:,160:160*2]).T
        # util.write_ply('test.ply',np.concatenate((pc,pc1)))
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        imgPCid[0] = np.stack((idx_s % 160, idx_s // 160)).T
        pointcloud[0,:3,:] = pc[idx_s,:].T
        pc_n = imgs_normal[0][:,160:160*2].reshape(-1, 3)
        pc_n = np.matmul(self.Rs[3][:3,:3].T, pc_n.T).T
        pointcloud[0,3:6,:] = pc_n[idx_s,:].T
        pc_c = imgs_rgb[0,:,160:160*2,:].reshape(-1,3)
        pointcloud[0,6:9,:] = pc_c[idx_s,::-1].T
        pc_s = imgs_s[0,:,160:160*2].reshape(-1)
        pointcloud[0,9:10,:] = pc_s[idx_s]

        pc = self.depth2pc(imgs_depth[1][:,160:160*2]).T
        idx_s = np.random.choice(range(len(pc)),self.num_points)
        imgPCid[1] = np.stack((idx_s % 160, idx_s // 160)).T
        pointcloud[1,:3,:] = pc[idx_s,:].T
        pc_n = imgs_normal[1][:,160:160*2].reshape(-1, 3)
        pc_n = np.matmul(self.Rs[3][:3,:3].T, pc_n.T).T
        pointcloud[1,3:6,:] = pc_n[idx_s,:].T
        pc_c = imgs_rgb[1,:,160:160*2,:].reshape(-1,3)
        pointcloud[1,6:9,:] = pc_c[idx_s,::-1].T
        pc_s = imgs_s[1,:, 160:160*2].reshape(-1)
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
        R_s2t = np.matmul(R[1], R_inv[0])
        pointcloud[0,:3,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,:3,:]) + R_s2t[:3,3:4]
        pointcloud[0,3:6,:] = np.matmul(R_s2t[:3,:3], pointcloud[0,3:6,:])
        
        #util.write_ply('test.ply', np.concatenate((pointcloud[0,:3,:].T,pointcloud[1,:3,:].T)),
        #  normal=np.concatenate((pointcloud[0,3:6,:].T,pointcloud[1,3:6,:].T)))
        if 1:
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
        
          rets['pos_s_360'] = (pos_s_360[None,:])
          rets['pos_t_360'] = (pos_t_360[None,:])
          rets['nor_s_360'] = (nor_s_360[None,:])
          rets['nor_t_360'] = (nor_t_360[None,:])

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
        
        
        Q = np.concatenate((util.rot2Quaternion(R_gt),t_gt.squeeze()))
        R_ = np.eye(4)
        R_[:3, :3] = R_gt
        R_[:3, 3] = t_gt.squeeze()
        R_inv = np.linalg.inv(R_)
        
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

        colors = np.random.rand(15+1,3)
        # resolution = 0.02 # 0.2m
        resolution = 0.04

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

        validind = np.where(mask)[0]
        invalidind = np.where(~mask)[0]
       
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
        dst = np.abs(((plane_eq_1[:3][None,:] * pc1).sum(1) + plane_eq_1[3]))
        mask = dst < 1.5 
        
        validind = np.where(mask)[0]
        invalidind = np.where(~mask)[0]
        
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




