import numpy as np
import io
try:
    from utils import train_op,torch_op
    import torch
    import torch.nn.functional as F
    from torch.autograd import Variable
    import torch.nn as nn
except:
    pass
from numpy.random import randn
import config
import sys
import collections
import cv2

import open3d as o3d
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
if sys.version_info[0] == 2:
    import Queue as queue
    string_classes = basestring
else:
    import queue
    string_classes = (str, bytes)
from sklearn.neighbors import KDTree

def point_cloud_overlap(pc_src,pc_tgt,R_gt_44):
    pc_src_trans = np.matmul(R_gt_44[:3,:3],pc_src.T) +R_gt_44[:3,3:4]
    tree = KDTree(pc_tgt)
    nearest_dist, nearest_ind = tree.query(pc_src_trans.T, k=1)
    nns2t = np.min(nearest_dist)
    hasCorres=(nearest_dist < 0.08)
    overlap_val_s2t = hasCorres.sum()/pc_src.shape[0]

    pc_tgt_trans = np.matmul(np.linalg.inv(R_gt_44),np.concatenate((pc_tgt.T,np.ones([1,pc_tgt.shape[0]]))))[:3,:]
    
    tree = KDTree(pc_src)
    nearest_dist, nearest_ind = tree.query(pc_tgt_trans.T, k=1)
    nnt2s = np.min(nearest_dist)
    hasCorres=(nearest_dist < 0.08)
    overlap_val_t2s = hasCorres.sum()/pc_tgt.shape[0]

    overlap_val = max(overlap_val_s2t,overlap_val_t2s)
    cam_dist_this = np.linalg.norm(R_gt_44[:3,3])
    pc_dist_this = np.linalg.norm(pc_src_trans.mean(1) - pc_tgt.T.mean(1))
    pc_nn = (nns2t+nnt2s)/2
    return overlap_val,cam_dist_this,pc_dist_this,pc_nn
def eldar():
    return os.path.exists('/scratch')
def parse_data(depth,rgb,norm,dataList,method):
    if 'suncg' in dataList or 'matterport' in dataList:
        #import pdb; pdb.set_trace()
        depth_src = depth[0,0,:,160:160*2]
        depth_tgt = depth[0,1,:,160:160*2]
        color_src = rgb[0,0,:,:,160:160*2].transpose(1,2,0)
        color_tgt = rgb[0,1,:,:,160:160*2].transpose(1,2,0)
        
        normal_src = norm[0,0,:,:,160:160*2].copy().transpose(1,2,0)
        normal_tgt = norm[0,1,:,:,160:160*2].copy().transpose(1,2,0)
        pc_src,mask_src = depth2pc(depth_src, dataList)
        pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)
        color_src = color_src.reshape(-1,3)[mask_src]/255.
        color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
        normal_src = normal_src.reshape(-1,3)[mask_src]
        normal_tgt = normal_tgt.reshape(-1,3)[mask_tgt]

    elif 'scannet' in dataList:
        if 'ours' in method:
            depth_src = depth[0,0,80-33:80+33,160+80-44:160+80+44]
            depth_tgt = depth[0,1,80-33:80+33,160+80-44:160+80+44]
            color_src = rgb[0,0,:,80-33:80+33,160+80-44:160+80+44].transpose(1,2,0)
            color_tgt = rgb[0,1,:,80-33:80+33,160+80-44:160+80+44].transpose(1,2,0)
            normal_src = norm[0,0,:,80-33:80+33,160+80-44:160+80+44].copy().transpose(1,2,0)
            normal_tgt = norm[0,1,:,80-33:80+33,160+80-44:160+80+44].copy().transpose(1,2,0)

            pc_src,mask_src = depth2pc(depth_src, dataList)
            pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)

            color_src = color_src.reshape(-1,3)[mask_src]/255.
            color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
            normal_src = normal_src.reshape(-1,3)[mask_src]
            normal_tgt = normal_tgt.reshape(-1,3)[mask_tgt]
            normal_src=normal_src/np.linalg.norm(normal_src,axis=1,keepdims=True)
            normal_tgt=normal_tgt/np.linalg.norm(normal_tgt,axis=1,keepdims=True)
            where_are_NaNs = np.isnan(normal_src.sum(1))
            normal_src[where_are_NaNs] = 0
            where_are_NaNs = np.isnan(normal_tgt.sum(1))
            normal_tgt[where_are_NaNs] = 0

        else:
            depth_src = depth[0,0,:,:]
            depth_tgt = depth[0,1,:,:]
            color_src = rgb[0,0,:,:].transpose(1,2,0)
            color_tgt = rgb[0,1,:,:].transpose(1,2,0)
            pc_src,mask_src = depth2pc(depth_src, dataList)
            pc_tgt,mask_tgt = depth2pc(depth_tgt, dataList)
            color_src = color_src.reshape(-1,3)[mask_src]/255.
            color_tgt = color_tgt.reshape(-1,3)[mask_tgt]/255.
            normal_src=None
            normal_tgt=None

    return depth_src,depth_tgt,normal_src,normal_tgt,color_src,color_tgt,pc_src,pc_tgt

def warping(view,R,dataList):
    if np.linalg.norm(R-np.eye(4)) == 0:
        return torch.zeros(view.shape).float().cuda()
    if 'suncg' in dataList:
        h=160
        colorpct=[]
        normalpct=[]
        depthpct=[]
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]
        
        for ii in range(4):
            colorpct.append(rgb[:,ii*h:(ii+1)*h,:].reshape(-1,3))
            normalpct.append(normal[:,ii*h:(ii+1)*h,:].reshape(-1,3))
            depthpct.append(depth[:,ii*h:(ii+1)*h].reshape(-1))
        colorpct=np.concatenate(colorpct)
        normalpct=np.concatenate(normalpct)
        depthpct=np.concatenate(depthpct)
        # get the coordinates of each point in the first coordinate system
        pct = Pano2PointCloud(depth,'suncg')# be aware of the order of returned pc!!!
        R_this_p=R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct,np.ones([1,pct.shape[1]]))))[:3,:]

        # assume always observe the second view(right view)
        colorpct=colorpct[h*h:h*h*2,:]
        depthpct=depthpct[h*h:h*h*2]
        normalpct=normalpct[h*h:h*h*2,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        pct_reproj=pct_reproj[:,h*h:h*h*2]
        
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        
        s2t_mask_p=(s2t_d_p!=0).astype('int')
        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)
    elif 'matterport' in dataList:
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]
        h=160
        pct,mask = depth2pc(depth[:,160:160*2],'matterport')# be aware of the order of returned pc!!!
        ii=1
        colorpct=rgb[:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask,:]
        normalpct=normal[:,ii*h:(ii+1)*h,:].reshape(-1,3)[mask,:]
        depthpct=depth[:,ii*h:(ii+1)*h].reshape(-1)[mask]
        # get the coordinates of each point in the first coordinate system
        R_this_p = R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        s2t_mask_p=(s2t_d_p!=0).astype('int')
        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)

    elif 'scannet' in dataList:
        assert(view.shape[2]==160 and view.shape[3]==640)
        h=view.shape[2]
        rgb=view[0,0:3,:,:].transpose(1,2,0)
        normal=view[0,3:6,:,:].transpose(1,2,0)
        depth=view[0,6,:,:]

        pct,mask = depth2pc(depth[80-33:80+33,160+80-44:160+80+44],'scannet')# be aware of the order of returned pc!!!
        colorpct = rgb[80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
        normalpct = normal[80-33:80+33,160+80-44:160+80+44,:].reshape(-1,3)[mask]
        depthpct = depth[80-33:80+33,160+80-44:160+80+44].reshape(-1)[mask]

        R_this_p=R
        pct_reproj = np.matmul(R_this_p,np.concatenate((pct.T,np.ones([1,pct.shape[0]]))))[:3,:]
        normalpct=np.matmul(R_this_p[:3,:3], normalpct.T).T
        s2t_rgb_p=reproj_helper(pct_reproj,colorpct,rgb.shape,'color',dataList)
        s2t_n_p=reproj_helper(pct_reproj,normalpct,rgb.shape,'normal',dataList)
        s2t_d_p=reproj_helper(pct_reproj,depthpct,rgb.shape[:2],'depth',dataList)
        s2t_mask_p=(s2t_d_p!=0).astype('int')

        view_s2t=np.expand_dims(np.concatenate((s2t_rgb_p,s2t_n_p,np.expand_dims(s2t_d_p,2),np.expand_dims(s2t_mask_p,2)),2),0).transpose(0,3,1,2)
    return view_s2t


def inverse(T):
    R, t = decompose(T)
    invT = np.zeros((4, 4))
    invT[:3, :3] = R.T
    invT[:3, 3] = -R.T.dot(t)
    invT[3, 3] = 1
    return invT

def pack(R, t):
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = t
    T[3, 3] = 1.0
    return T

def decompose(T):
    R = T[:3, :3]
    t = T[:3, 3]
    return R, t
def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    if R_hat.shape == (3,3):
        R_hat = R_hat[np.newaxis,:]
    if R.shape == (3,3):
        R = R[np.newaxis,:]
    n = R.shape[0]
    trace_idx = [0,4,8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def apply_quaternion(Q, pc, normal=False):
    # Q: [b, 7]
    # pc: [b, n, 3]
    # x: [b, n, c]
    
    Q_expand = Q.unsqueeze(1).repeat(1, pc.shape[1], 1)
    x = qrot(Q_expand[:,:,:4], pc)
    if not normal:
        x += Q_expand[:, :, 4:]
    return x

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #m.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def read_super4pcs_mat(path):
    with open(path, 'r') as f:
        lines = f.readlines()[2:]
        r = np.zeros([4,4])
        for i in range(4):
            line = lines[i]
            tmp = line.strip('\n ').split('  ')
            tmp = [float(v) for v in tmp]
            r[i,:] = tmp
        return r
def rnd_color(pc):
    return np.tile(np.random.rand(3)[None,:], [pc.shape[0], 1])
def perturb(pc):
    return pc + np.random.rand(*pc.shape)*1e-4
def pcloud_point(x,color=None,normal=None,eps=1e-3):
  pcloud = np.tile(x[None,:], [100,1])
  pcloud += (np.random.rand(*pcloud.shape)-0.5)*eps
  if color is None:
    pcolor = np.tile(np.array([0,1,0])[None,:],[pcloud.shape[0],1])
  else:
    assert(len(color) == 3)
    pcolor = np.tile(np.array(color)[None,:],[pcloud.shape[0],1])
  if normal is None:
    pnormal = np.tile(np.array([0,1,0])[None,:],[pcloud.shape[0],1])
  else:
    assert(len(normal) == 3)
    pnormal = np.tile(np.array(normal)[None,:],[pcloud.shape[0],1])   
  return pcloud, pcolor,pnormal
def pcloud_line(prev,cur,color=None):
  alpha = np.linspace(0,1,100)
  pcloud = prev[None,:] + alpha[:,None] * (cur - prev)[None,:]
  if color is None:
    pcolor = np.tile(np.array([0,1,0])[None,:],[pcloud.shape[0],1])
  else:
    assert(len(color) == 3)
    pcolor = np.tile(np.array(color)[None,:],[pcloud.shape[0],1])
  return pcloud, pcolor
def precision_recall(x, y, fp='test.png', THRESH=np.array([0.7,0.7,0.7])):
    mARs = []
    mAPs = []
    #for thresh in np.linspace(0, 0.99, 20):
    for thresh in np.linspace(0.5, 0.99, 20):
        pred = np.zeros([len(x)])

        #THRESH_THIS = THRESH + (1 - THRESH) * thresh
        #tp = (x[:,1:] > THRESH_THIS[None, :])
        #pred[tp.sum(1)==1] = np.argmax(tp,1)[tp.sum(1)==1]+1
        
        mask = (np.max(x[:,1:], 1) < thresh)
        pred[mask] = 0
        pred[~mask] = np.argmax(x[~mask,1:], 1)+1
        
        # average precision 
        mAP = []
        for j in range(1,4):
            if (pred==j).sum():
                mAP.append((y[pred == j] == j).mean())
        mAP = np.mean(mAP)
        if np.isnan(mAP):
            mAP = 0
        
        # average recall 
        mAR = []
        for j in range(1,4):
            if (y==j).sum():
                mAR.append((pred[y == j] == j).mean())
        mAR = np.mean(mAR)
        if np.isnan(mAR):
            mAR = 0
        mARs.append(mAR)
        mAPs.append(mAP)
        print(mAP, mAR)
    plt.clf()
    plt.plot(mARs, mAPs)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.savefig(fp)
def vis_relation(pc1, pc2, pairID1, pairID2, rel, n=50):
    pc_gt = np.concatenate((pc1, pc2))
    pt_add = []
    pt_add_c = []
    for _ in range(n):
        idx = np.random.choice(range(rel.shape[0]), 1)[0]
        pt0 = pc1[pairID1[idx]]
        pt1 = pc2[pairID2[idx]]
        pt_add_this = []
        pt_add_this_c = []
        if rel[idx] == 1:
            pt_add_this, pt_add_this_c = pcloud_line(pt0,pt1,color=np.array([1,0,0]))
        elif rel[idx] == 2:
            pt_add_this, pt_add_this_c = pcloud_line(pt0,pt1,color=np.array([0,1,0]))
        elif rel[idx] == 3:
            pt_add_this, pt_add_this_c = pcloud_line(pt0,pt1,color=np.array([0,0,1]))
        elif rel[idx] == 0:
            pt_add_this, pt_add_this_c = pcloud_line(pt0,pt1,color=np.array([0,0,0]))
        if len(pt_add_this):
            pt_add.append(pt_add_this)
            pt_add_c.append(pt_add_this_c)
    colors = np.concatenate((np.tile(np.array([1,1,0])[None, :], [pc_gt.shape[0]//2, 1]),
            np.tile(np.array([0,1,1])[None, :], [pc_gt.shape[0]//2, 1])))
    pt_add = np.concatenate(pt_add)
    pt_add_c = np.concatenate(pt_add_c)
    write_ply('test.ply', point=np.concatenate((pc_gt, pt_add)), color=np.concatenate((colors, pt_add_c)))


def write_ply(fn, point, normal=None, color=None):
  
  ply = o3d.geometry.PointCloud()
  ply.points = o3d.utility.Vector3dVector(point)
  if color is not None:
    ply.colors = o3d.utility.Vector3dVector(color)
  if normal is not None:
    ply.normals = o3d.utility.Vector3dVector(normal)
  o3d.io.write_point_cloud(fn, ply)

def line(point_s, point_t, resol=0.005,color=(1,0,1)):
  '''
  point_s: [k, 3]
  point_t: [k, 3]
  '''
  n = point_s.shape[0]
  line_v = []
  line_c = []
  for i in range(n):
    dst = np.linalg.norm((point_t[i] - point_s[i]))
    alpha = np.linspace(0,1,dst/resol)
    line_v.append(point_s[i][None,:] + (point_t[i] - point_s[i])[None,:]*alpha[:,None])
    line_c.append(np.tile(np.array(color)[None,:],[len(alpha),1]))
  line_v = np.concatenate(line_v)
  line_c = np.concatenate(line_c)
  return line_v, line_c

def apply_mask(x,maskMethod,*arg):
    # input: [n,c,h,w]
    h=x.shape[2]
    w=x.shape[3]
    tp = np.zeros([x.shape[0],1,x.shape[2],x.shape[3]])
    geow=np.zeros([x.shape[0],1,x.shape[2],x.shape[3]])
    if maskMethod == 'second':
        tp[:,:,:h,h:2*h]=1
        ys,xs=np.meshgrid(range(h),range(w),indexing='ij')
        dist=np.stack((np.abs(xs-h),np.abs(xs-(2*h)),np.abs(xs-w-h),np.abs(xs-w-(2*h))),0)
        dist=dist.min(0)/h
        sigmaGeom=0.7
        dist=np.exp(-dist/(2*sigmaGeom**2))
        dist[:,h:2*h]=0
        geow = torch_op.v(np.tile(np.reshape(dist,[1,1,dist.shape[0],dist.shape[1]]),[geow.shape[0],1,1,1]))
    elif maskMethod == 'kinect':
        assert(w==640 and h==160)
        dw = int(89.67//2) # 44
        dh = int(67.25//2) # 33
        tp[:,:,80-dh:80+dh,160+80-dw:160+80+dw]=1
        geow = 1-torch_op.v(tp)
    elif maskMethod == '120fov':
      tp[:,:,100-48:100+48,200-64:200+64]=1
      geow = tp.copy()*20
      geow[tp==0]=1
      geow = torch_op.v(geow)
    tp=torch_op.v(tp)
    x=x*tp
    return x,tp,geow

def randomRotation(epsilon):
    axis=(np.random.rand(3)-0.5)
    axis/=np.linalg.norm(axis)
    dtheta=np.random.randn(1)*np.pi*epsilon
    K=np.array([0,-axis[2],axis[1],axis[2],0,-axis[0],-axis[1],axis[0],0]).reshape(3,3)
    dR=np.eye(3)+np.sin(dtheta)*K+(1-np.cos(dtheta))*np.matmul(K,K)
    return dR

def horn87_v1(src, tgt, weight=None): # does not substract center, compare to horn87_np_v2
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src.unsqueeze(0), tgt.unsqueeze(0)
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = torch.ones(k,1,nPts).cuda().float()
    else:
        has_weight=True
        weight = weight.view(k,1,nPts)
    weight = weight / weight.sum(2,keepdim=True)
    src_ = src
    tgt_ = tgt
    if has_weight:
        for i in range(k):
            tgt_[i] *= weight[i]

    H = torch.bmm(src_, tgt_.transpose(2,1))
    R_ret = []
    for i in range(k):
        try:
            u, s, v = torch.svd(H[i,:,:].cpu())
            R = torch.matmul(v, u.t())
            det = torch.det(R)
            if det < 0:
                R = torch.matmul(v, torch.matmul(torch.diagflat(torch.FloatTensor([1,1,-1])),u.t()))
            R_ret.append(R.view(-1,3,3))

        except:
            print('rigid transform failed to converge')
            print('H:{}'.format(torch_op.npy(H)))

            R_ret.append(Variable(torch.eye(3).view(1,3,3), requires_grad=True))
    
    R_ret = torch.cat(R_ret).cuda()

    return R_ret

def horn87_np_v2(src, tgt,weight=None):
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src[np.newaxis,:], tgt[np.newaxis,:]
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = np.ones([k,1,nPts])
    else:
        has_weight=True
        weight = weight.reshape(k,1,nPts)

    src_ = src
    tgt_ = tgt

    if has_weight:
        tgt_ = tgt_.copy()
        for i in range(k):
            tgt_[i] *= weight[i]
    M = np.matmul(src_, tgt_.transpose(0,2,1))
    R_ret = []
    for i in range(k):
        N = np.array([[M[i,0, 0] + M[i,1, 1] + M[i,2, 2], M[i,1, 2] - M[i,2, 1], M[i,2, 0] - M[i,0, 2], M[i,0, 1] - M[i,1, 0]], 
                        [M[i,1, 2] - M[i,2, 1], M[i,0, 0] - M[i,1, 1] - M[i,2, 2], M[i,0, 1] + M[i,1, 0], M[i,0, 2] + M[i,2, 0]], 
                        [M[i,2, 0] - M[i,0, 2], M[i,0, 1] + M[i,1, 0], M[i,1, 1] - M[i,0, 0] - M[i,2, 2], M[i,1, 2] + M[i,2, 1]], 
                        [M[i,0, 1] - M[i,1, 0], M[i,2, 0] + M[i,0, 2], M[i,1, 2] + M[i,2, 1], M[i,2, 2] - M[i,0, 0] - M[i,1, 1]]])
        v, u = np.linalg.eig(N)
        id = v.argmax()

        q = u[:, id]
        R_ret.append(np.array([[q[0]**2+q[1]**2-q[2]**2-q[3]**2, 2*(q[1]*q[2]-q[0]*q[3]), 2*(q[1]*q[3]+q[0]*q[2])], 
                        [2*(q[2]*q[1]+q[0]*q[3]), q[0]**2-q[1]**2+q[2]**2-q[3]**2, 2*(q[2]*q[3]-q[0]*q[1])], 
                        [2*(q[3]*q[1]-q[0]*q[2]), 2*(q[3]*q[2]+q[0]*q[1]), q[0]**2-q[1]**2-q[2]**2+q[3]**2]]).reshape(1,3,3))
    R_ret = np.concatenate(R_ret)
    return R_ret
def horn87_np(src, tgt, weight=None):
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    #   (R, t) ([(k),3,3], [(k),3,1])
    '''
    if len(src.shape) == 2 and len(tgt.shape) == 2:
        src, tgt = src[np.newaxis,:], tgt[np.newaxis,:]
    assert(src.shape[2] == tgt.shape[2])
    nPts = src.shape[2]
    k = src.shape[0]
    has_weight=False
    if weight is None:
        weight = np.ones([k,1,nPts])
    else:
        has_weight=True
        weight = weight.reshape(k,1,nPts)
    #weight = weight / weight.sum(2,keepdims=True)
    src_c = (src * weight).sum(2,keepdims=True) / weight.sum(2,keepdims=True)
    tgt_c = (tgt * weight).sum(2,keepdims=True) / weight.sum(2,keepdims=True)
    src_ = src - src_c
    tgt_ = tgt - tgt_c

    
    if has_weight:
        tgt_ = tgt_.copy()
        for i in range(k):
            tgt_[i] *= weight[i]
    
    H = np.matmul(src_, tgt_.transpose(0,2,1))
    
    
    R_ret = []
    for i in range(k):
        try:
            u, s, vh = np.linalg.svd(H[i,:,:])
            R = np.matmul(u, vh).T
            det = np.linalg.det(R)
            if det < 0:
                R = np.matmul(u, np.matmul(np.diag([1,1,-1]),vh)).T
            R_ret.append(R.reshape(-1,3,3))
            ti = -np.matmul(R, src_c[i]) + tgt_c[i]

        except:
            print('rigid transform failed to converge')
            print('H:{}'.format(H))

            R_ret.append(np.eye(3).reshape(1,3,3))
    
    R_ret = np.concatenate(R_ret)
    t = -np.matmul(R_ret, src_c) + tgt_c

    return R_ret, t
def drawMatch(img0,img1,src,tgt,color=['b']):
    if len(img0.shape)==2:
      img0=np.expand_dims(img0,2)
    if len(img1.shape)==2:
      img1=np.expand_dims(img1,2)
    h,w = img0.shape[0],img0.shape[1]
    img = np.zeros([2*h,w,3])
    img[:h,:,:] = img0
    img[h:,:,:] = img1
    n = len(src)
    colors=[]
    if len(color)!=1:
        assert(len(color) == n)
        for i in range(n):
            if color[i] == 'b':
                colors.append((255,0,0))
            elif color[i] == 'r':
                colors.append((0,0,255))
    else:
        for i in range(n):
            if color[0] == 'b':
                colors.append((255,0,0))
            elif color[0] == 'r':
                colors.append((0,0,255))
    for i in range(n):
      cv2.circle(img, (int(src[i,0]), int(src[i,1])), 3,colors[i],-1)
      cv2.circle(img, (int(tgt[i,0]), int(tgt[i,1])+h), 3,colors[i],-1)
      cv2.line(img, (int(src[i,0]),int(src[i,1])),(int(tgt[i,0]),int(tgt[i,1])+h),colors[i],3)
    return img

def drawKeypoint(imgSize, pts):
    # imgSize: [h,w]
    # pts: [n,2]
    ret=np.zeros(imgSize)
    color=(255,0,0)
    for i in range(len(pts)):
        cv2.circle(ret,(int(pts[i,0]), int(pts[i,1])), 3,color,-1)
    return ret

def normalize(v, tolerance=0.00001):
    mag2 = sum(n * n for n in v)
    if abs(mag2 - 1.0) > tolerance:
        mag = sqrt(mag2)
        v = tuple(n / mag for n in v)
    return v

def q_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return w, x, y, z

def qv_mult(q1, v1):
    q2 = (0.0,) + v1
    return q_mult(q_mult(q1, q2), q_conjugate(q1))[1:]
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]
    
    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)
    
    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)
def q_conjugate(q):
    w, x, y, z = q
    return (w, -x, -y, -z)

def axisangle_to_q(v, theta):
    v = normalize(v)
    x, y, z = v
    theta /= 2
    w = cos(theta)
    x = x * sin(theta)
    y = y * sin(theta)
    z = z * sin(theta)
    return w, x, y, z

def q_to_axisangle(q):
    w, v = q[0], q[1:]
    theta = acos(w) * 2.0
    return normalize(v), theta

def rotmat_to_axisangle_np(rotation):
    # rotation: [n, 3, 3]
    # return: [n, 3]
    n = rotation.shape[0]
    trace_idx = [0,4,8]
    trace = rotation.reshape(n,-1)[:,trace_idx].sum(1)
    theta = np.arccos(((trace - 1)/2).clip(-1,1))
    axis = np.zeros([rotation.shape[0],3])
    epsilon = 1e-6
    singular_index = (np.abs(rotation[:,0,1]-rotation[:,1,0])<epsilon) * \
        (np.abs(rotation[:,0,2]-rotation[:,2,0])< epsilon) * \
        (np.abs(rotation[:,1,2]-rotation[:,2,1])< epsilon)
    singular_index_0 = (np.abs(rotation[:,0,1]+rotation[:,1,0])<epsilon) * \
        (np.abs(rotation[:,0,2]+rotation[:,2,0])< epsilon) * \
        (np.abs(rotation[:,1,2]+rotation[:,2,1])< epsilon) * \
        (np.abs(rotation[:,0,0]+rotation[:,1,1]+rotation[:,2,2]-3)< epsilon)
    singular_index_180 = singular_index * (1-singular_index_0)
    
	#return new axisAngle(0,1,0,0); // zero angle, arbitrary axis
	

    axis[:, 0] = rotation[:,2,1] - rotation[:,1,2]
    axis[:, 1] = rotation[:,0,2] - rotation[:,2,0]
    axis[:, 2] = rotation[:,1,0] - rotation[:,0,1]
    axis = axis / (2 * np.sin(theta) + 1e-6).reshape(-1,1)
    
    axis = axis*theta.reshape(-1,1)
    
    axis[np.where(singular_index_0)] = np.array([1,0,0])*np.pi*2
    return axis

def angle_axis_to_rotation_matrix(angle_axis_with_t):
    """Convert 3d vector of axis-angle rotation to 4x4 rotation matrix

    Args:
        angle_axis (Tensor): tensor of 3d vector of axis-angle rotations.

    Returns:
        Tensor: tensor of 4x4 rotation matrices.

    Shape:
        - Input: :math:`(N, 3)`
        - Output: :math:`(N, 4, 4)`

    Example:
        >>> input = torch.rand(1, 3)  # Nx3
        >>> output = tgm.angle_axis_to_rotation_matrix(input)  # Nx4x4
    """
    def _compute_rotation_matrix(angle_axis, theta2, eps=1e-6):
        # We want to be careful to only evaluate the square root if the
        # norm of the angle_axis vector is greater than zero. Otherwise
        # we get a division by zero.
        k_one = 1.0
        theta = torch.sqrt(theta2)
        wxyz = angle_axis / (theta + eps)
        wx, wy, wz = torch.chunk(wxyz, 3, dim=1)
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        r00 = cos_theta + wx * wx * (k_one - cos_theta)
        r10 = wz * sin_theta + wx * wy * (k_one - cos_theta)
        r20 = -wy * sin_theta + wx * wz * (k_one - cos_theta)
        r01 = wx * wy * (k_one - cos_theta) - wz * sin_theta
        r11 = cos_theta + wy * wy * (k_one - cos_theta)
        r21 = wx * sin_theta + wy * wz * (k_one - cos_theta)
        r02 = wy * sin_theta + wx * wz * (k_one - cos_theta)
        r12 = -wx * sin_theta + wy * wz * (k_one - cos_theta)
        r22 = cos_theta + wz * wz * (k_one - cos_theta)
        rotation_matrix = torch.cat(
            [r00, r01, r02, r10, r11, r12, r20, r21, r22], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    def _compute_rotation_matrix_taylor(angle_axis):
        rx, ry, rz = torch.chunk(angle_axis, 3, dim=1)
        k_one = torch.ones_like(rx)
        rotation_matrix = torch.cat(
            [k_one, -rz, ry, rz, k_one, -rx, -ry, rx, k_one], dim=1)
        return rotation_matrix.view(-1, 3, 3)

    # stolen from ceres/rotation.h
    angle_axis = angle_axis_with_t[:, :3]
    _angle_axis = torch.unsqueeze(angle_axis, dim=1)
    theta2 = torch.matmul(_angle_axis, _angle_axis.transpose(1, 2))
    theta2 = torch.squeeze(theta2, dim=1)

    # compute rotation matrices
    rotation_matrix_normal = _compute_rotation_matrix(angle_axis, theta2)
    rotation_matrix_taylor = _compute_rotation_matrix_taylor(angle_axis)

    # create mask to handle both cases
    eps = 1e-6
    mask = (theta2 > eps).view(-1, 1, 1).to(theta2.device)
    mask_pos = (mask).type_as(theta2)
    mask_neg = (mask == False).type_as(theta2)  # noqa

    # create output pose matrix
    batch_size = angle_axis.shape[0]
    rotation_matrix = torch.eye(4).to(angle_axis.device).type_as(angle_axis)
    rotation_matrix = rotation_matrix.view(1, 4, 4).repeat(batch_size, 1, 1)
    # fill output matrix with masked values
    rotation_matrix[..., :3, :3] = \
        mask_pos * rotation_matrix_normal + mask_neg * rotation_matrix_taylor
    
    rotation_matrix[:, :3, 3] = angle_axis_with_t[:, 3:6]
    return rotation_matrix  # Nx4x4


def axisangle_to_rotmat(axisangle):
    # pose: [n, 6]
    # return R: [n, 4, 4]
    n = axisangle.size(0)
    v = axisangle[:,:3]
    #theta = pose[:, 3]
    epsilon = 0.000000001   # used to handle 0/0
    v_length = torch.sqrt(torch.sum(v*v, dim=1))
    vx = (v[:, 0] + epsilon) / (v_length + epsilon)
    vy = (v[:, 1] + epsilon) / (v_length + epsilon)
    vz = (v[:, 2] + epsilon) / (v_length + epsilon)
    
    """
    if 1:
      m = Variable(torch.zeros(n, 3, 3).cuda(), requires_grad=True)
      m[:, 0, 1] = -vz
      m[:, 0, 2] = vy
      m[:, 1, 0] = vz
      m[:, 1, 2] = -vx
      m[:, 2, 0] = -vy
      m[:, 2, 1] = vx
    """
    
    zero_ = torch.zeros_like(vx)
    m = torch.stack([zero_, -vz, vy, vz, zero_, -vx, -vy, vx, zero_]).transpose(0,1).view(n, 3, 3)

    I3 = Variable(torch.eye(3).view(1,3,3).repeat(n,1,1).cuda())
    R = Variable(torch.eye(4).view(1,4,4).repeat(n,1,1).cuda())
    R[:,:3,:3] = I3 + torch.sin(v_length).view(n,1,1)*m + (1-torch.cos(v_length)).view(n,1,1)*torch.bmm(m, m)
    R[:,:3,3] = R[:,:3,3] + axisangle[:,3:]
    return R
def rot2Quaternion(rot):
    # rot: [3,3]
    assert(rot.shape==(3,3))
    tr = np.trace(rot)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qz = (rot[2,1]-rot[1,2]) / S
        qy = (rot[0,2]-rot[2,0]) / S
        qx = (rot[1,0]-rot[0,1]) / S
    elif (rot[0,0]>rot[1,1]) and (rot[0,0]>rot[2,2]):
        S = np.sqrt(1.0 + rot[0,0] - rot[1,1] - rot[2,2]) * 2
        qw = (rot[2,1] - rot[1,2]) / S
        qz = 0.25 * S
        qy = (rot[0,1]+rot[1,0]) / S
        qx = (rot[0,2]+rot[2,0]) / S
    elif rot[1,1] > rot[2,2]:
        S = np.sqrt(1.0 + rot[1,1] - rot[0,0] - rot[2,2]) * 2
        qw = (rot[0,2] - rot[2,0]) / S
        qz = (rot[0,1] + rot[1,0]) / S
        qy = 0.25 * S
        qx = (rot[1,2]+rot[2,1]) / S
    else:
        S = np.sqrt(1.0 + rot[2,2] - rot[0,0] - rot[1,1]) * 2
        qw = (rot[1,0] - rot[0,1]) / S
        qz = (rot[0,2] + rot[2,0]) / S
        qy = (rot[1,2] + rot[2,1]) / S
        qx = 0.25 * S

    return np.array([qw, qz, qy, qx])

def quaternion2Rot(q):
    # q:[n, 7]
    # R:[n, 4, 4]
    R00 = q[:, 0].pow(2) + q[:, 1].pow(2) - q[:, 2].pow(2) - q[:, 3].pow(2)
    R01 = 2 * (q[:, 1]*q[:, 2] - q[:, 0] * q[:, 3])
    R02 = 2 * (q[:, 0] * q[:, 2] + q[:, 1] * q[:, 3])
    R10 = 2 * (q[:, 1] * q[:, 2] + q[:, 0] * q[:, 3])
    R11 = q[:, 0].pow(2) - q[:, 1].pow(2) + q[:, 2].pow(2) - q[:, 3].pow(2)
    R12 = 2 * (q[:, 2] * q[:, 3] - q[:, 0] * q[:, 1])
    R20 = 2 * (q[:, 1] * q[:, 3] - q[:, 0] * q[:, 2])
    R21 = 2 * (q[:, 0] * q[:, 1] + q[:, 2] * q[:, 3])
    R22 = q[:, 0].pow(2) - q[:, 1].pow(2) - q[:, 2].pow(2) + q[:, 3].pow(2)
    R03 = q[:, 4]
    R13 = q[:, 5]
    R23 = q[:, 6]
    R30 = torch.zeros([q.shape[0]]).float().cuda()
    R31 = torch.zeros([q.shape[0]]).float().cuda()
    R32 = torch.zeros([q.shape[0]]).float().cuda()
    R33 = torch.ones([q.shape[0]]).float().cuda()

    R = torch.stack((R00, R01, R02, R03, R10, R11, R12, R13, R20, R21, R22, R23, R30, R31, R32, R33), -1).view(-1, 4, 4)
    return R

def quaternion2Rot_np(q):
    # q:[4]
    # R:[3,3]
    R = np.zeros([3,3])
    R[0,0] = q[0]**2 + q[1]**2 - q[2]**2 - q[3]**2
    R[0,1] = 2*(q[1]*q[2] - q[0]*q[3])
    R[0,2] = 2*(q[0]*q[2] + q[1]*q[3])
    R[1,0] = 2*(q[1]*q[2] + q[0]*q[3])
    R[1,1] = q[0]**2 - q[1]**2 + q[2]**2 - q[3]**2
    R[1,2] = 2*(q[2]*q[3] - q[0]*q[1])
    R[2,0] = 2*(q[1]*q[3] - q[0]*q[2])
    R[2,1] = 2*(q[0]*q[1]+q[2]*q[3])
    R[2,2] = q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    return R

def Rnd(x):
  return max(-2 * x, min(2 * x, randn() * x))
  
def Flip(img):
  if len(img.shape) == 3:
    return img[:, :, ::-1].copy()  
  elif len(img.shape) == 4:
    return img[:, :, :, ::-1].copy()  
  else:
    raise Exception('Flip shape error')
def transform4x4(pc, T):
  # T: [4,4]
  # pc: [n, 3]
  # return: [n, 3]
  return (np.matmul(T[:3,:3], pc.T) + T[:3, 3:4]).T
def compose_src_tgt(pc_s, pc_t, T):
    pc_src_tp  = (np.matmul(T[:3,:3], pc_s.T) + T[:3, 3:4]).T
    return np.concatenate((pc_src_tp, pc_t))
def depth2pc(depth,dataList):
    
    if 'suncg' in dataList:

        w,h = depth.shape[1], depth.shape[0]
        assert(w == 160 and h == 160)
        #Rs = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        xs=xs.flatten()[mask]*zs
        ys=ys.flatten()[mask]*zs
        pc = np.stack((xs,ys,-zs),1)
        # assume second view!
        #pc=np.matmul(Rs[:3,:3],pc.T).T
    elif 'matterport' in dataList:
        w,h = depth.shape[1], depth.shape[0]
        assert(w == 160 and h == 160)
        #Rs = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        # transform from ith frame to 0th frame
        ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
        zs = depth.flatten()
        mask = (zs!=0)
        zs = zs[mask]
        xs=xs.flatten()[mask]*zs
        ys=ys.flatten()[mask]*zs
        pc = np.stack((xs,ys,-zs),1)

    elif 'scannet' in dataList:
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
        elif (depth.shape[0] == 66 and depth.shape[1] == 88):
            w,h = depth.shape[1], depth.shape[0]
            # transform from ith frame to 0th frame
            ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
            ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
            zs = depth.flatten()
            mask = (zs!=0)
            zs = zs[mask]
            xs=xs.flatten()[mask]*zs
            ys=ys.flatten()[mask]*zs
            pc = np.stack((xs*w/160,ys*h/160,-zs),1)
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
            

    elif 'scan_pano' in dataList:
        w,h = depth.shape[1], depth.shape[0]
        assert(w == 512 and h == 256)
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')

        lat = (xs / w) * 2 * np.pi - np.pi
        long_ = -(ys/h*np.pi - np.pi/2)

        r = depth.flatten()
        mask = (r>0)
        r = r[mask]
        long_ = long_.flatten()[mask]
        lat = lat.flatten()[mask]
       
        zs = r * np.sin(long_)
        xs = r * np.cos(long_) * np.sin(lat)
        ys = r * np.cos(long_) * np.cos(lat)

        pc = np.concatenate((xs, zs, -ys)).reshape(3,-1)


    return pc,mask

def PanoIdx(index,h,w):
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

def reproj_helper(pct,colorpct,out_shape,mode,dataList):
    if 'suncg' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        # find which plane they intersect with
        h=out_shape[0]
        tp=pct.copy()
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

        tp=np.matmul(Rs[1][:3,:3].T,pct)
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

        tp=np.matmul(Rs[2][:3,:3].T,pct)
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

        tp=np.matmul(Rs[3][:3,:3].T,pct)
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
    elif 'matterport' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        h=out_shape[0]
        tp=np.matmul(Rs[3][:3,:3].T,pct)
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

        tp=np.matmul(Rs[0][:3,:3].T,pct)
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

        tp=np.matmul(Rs[1][:3,:3].T,pct)
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

        tp=np.matmul(Rs[2][:3,:3].T,pct)
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
    elif 'scannet' in dataList:
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        h=out_shape[0]
        tp=np.matmul(Rs[3][:3,:3].T,pct)
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

        tp=np.matmul(Rs[0][:3,:3].T,pct)
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

        tp=np.matmul(Rs[1][:3,:3].T,pct)
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

        tp=np.matmul(Rs[2][:3,:3].T,pct)
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
    
def Pano2PointCloud(depth,dataList):
    # The order of rendered 4 view are different between suncg and scannet/matterport. 
    # Hacks to separately deal with each dataset and get the corrected assembled point cloud.
    # TODO: FIX THE DATASET INCONSISTENCY
    if 'suncg' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
            zs = depth[:,i*w:(i+1)*w].flatten()
            ys_this, xs_this = ys.flatten()*zs, xs.flatten()*zs
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[i][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    elif 'matterport' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
            zs = depth[:,i*w:(i+1)*w].flatten()
            ys_this, xs_this = ys.flatten()*zs, xs.flatten()*zs
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[(i-1)%4][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    elif 'scannet' in dataList:
        assert(depth.shape[0]==160 and depth.shape[1]==640)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        w,h = depth.shape[1]//4, depth.shape[0]
        ys, xs = np.meshgrid(range(h),range(w),indexing='ij')
        ys, xs = (0.5-ys / h)*2, (xs / w-0.5)*2
        pc = []
        for i in range(4):
           
            zs = depth[:,i*w:(i+1)*w].flatten()
            mask=(zs!=0)
            zs=zs[mask]
            # ys_this, xs_this = ys.flatten()[mask]*zs/(1.1895*2), xs.flatten()[mask]*zs/(0.8921875*2)
            ys_this, xs_this = ys.flatten()[mask]*zs, xs.flatten()[mask]*zs
            pc_this = np.concatenate((xs_this,ys_this,-zs)).reshape(3,-1) # assume depth clean
            pc_this = np.matmul(Rs[(i-1)%4][:3,:3],pc_this)
            pc.append(pc_this)
        pc = np.concatenate(pc,1)
    return pc

def saveimg(kk,filename='test.png'):
    cv2.imwrite(filename,(kk-kk.min())/(kk.max()-kk.min())*255)

def pnlayer(depth,normal,plane,dataList,representation):
    # dp: [n,1,h,w]
    # n: [n,3,h,w]
    if 'suncg' in dataList or 'matterport' in dataList:
        n,h,w = depth.shape[0],depth.shape[2],depth.shape[3]
        assert(h==w//4)
        Rs = np.zeros([4,4,4])
        Rs[0] = np.eye(4)
        Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
        Rs=torch_op.v(Rs)
        loss_pn=0
        for i in range(4):
            plane_this=plane[:,0,:,i*h:(i+1)*h].contiguous()
            depth_this=depth[:,0,:,i*h:(i+1)*h].contiguous()
            ys, xs = np.meshgrid(range(h),range(h),indexing='ij')
            ys, xs = (0.5-ys / h)*2, (xs / h-0.5)*2
            xs = xs.flatten()
            ys = ys.flatten()
            zs = plane_this.view(-1)
            mask = (zs!=0)
            masknpy = torch_op.npy(mask)
            normal_this=normal[:,:,:,i*h:(i+1)*h].permute(0,2,3,1).contiguous().view(-1,3)
            if 'suncg' in dataList:
                normal_this=torch.matmul(Rs[i][:3,:3].t(),normal_this.t()).t()
            elif 'matterport' in dataList:
                normal_this=torch.matmul(Rs[(i-1)%4][:3,:3].t(),normal_this.t()).t()
            ray = np.tile(np.stack((-xs[masknpy],-ys[masknpy],np.ones(len(xs))),1),[n,1])
            ray = torch_op.v(ray)
            pcPn=(zs/(ray*normal_this+1e-6).sum(1)).unsqueeze(1)*ray
            
            xs=torch_op.v(np.tile(xs,n))
            ys=torch_op.v(np.tile(ys,n))
            zs=depth_this.view(-1)

            xs=xs*zs
            ys=ys*zs
            pcD = torch.stack((xs,ys,-zs),1)
            loss_pn+=(pcD-pcPn).clamp(-5,5).abs().mean()
    elif 'scannet' in dataList:
        raise Exception("not implemented: scannet/skybox representation")

    return loss_pn

def COSINELoss(input, target):
    loss = (1-(input.view(-1,3)*target.view(-1,3)).sum(1)).pow(2).mean()
    return loss

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    
def collate_fn_cat(batch):
  "Puts each data field into a tensor with outer dimension batch size"
  if torch.is_tensor(batch[0]):
    out = None
    return torch.cat(batch, 0, out=out)
    # for rnn variable length input
    """
    elif type(batch[0]).__name__ == 'list':
        import ipdb;ipdb.set_trace()
        dim = 0
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[dim], batch))
        # pad according to max_len
        #batch = map(lambda (x, y):(pad_tensor(x, pad=max_len, dim=dim), y), batch)
        # stack all
        xs = torch.stack(map(lambda x: x[0], batch), dim=0)
        ys = torch.LongTensor(map(lambda x: x[1], batch))
    """
  elif type(batch[0]).__module__ == 'numpy':
    elem = batch[0]
    if type(elem).__name__ == 'ndarray':
      """
      seq_length = np.array([b.shape[0] for b in batch])
      if (seq_length != seq_length.mean()).sum() > 0: # need paddding
        import ipdb;ipdb.set_trace()
        max_len = max(seq_length)
        perm = np.argsort(seq_length)[::-1]
        batch = batch[perm]
        return torch.stack(map(lambda x: pad_tensor(x, max_len, 0), batch))
      """
      try:
        torch.cat([torch.from_numpy(b) for b in batch], 0)
      except:
        return batch
        import ipdb;ipdb.set_trace()
      return torch.cat([torch.from_numpy(b) for b in batch], 0)
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      numpy_type_map = {
        'float64': torch.DoubleTensor,
        'float32': torch.FloatTensor,
        'float16': torch.HalfTensor,
        'int64': torch.LongTensor,
        'int32': torch.IntTensor,
        'int16': torch.ShortTensor,
        'int8': torch.CharTensor,
        'uint8': torch.ByteTensor,
       }
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  elif isinstance(batch[0], int):
    return torch.LongTensor(batch)
  elif isinstance(batch[0], float):
    return torch.DoubleTensor(batch)
  elif isinstance(batch[0], string_classes):
    return batch
  elif isinstance(batch[0], collections.Mapping):
    return {key: collate_fn_cat([d[key] for d in batch]) for key in batch[0]}
  elif isinstance(batch[0], collections.Sequence):
    transposed = zip(*batch)
    return [collate_fn_cat(samples) for samples in transposed]

  raise TypeError(("batch must contain tensors, numbers, dicts or lists; found {}"
                     .format(type(batch[0]))))

def Rz(psi):
    m = np.zeros([3, 3])
    m[0, 0] = np.cos(psi)
    m[0, 1] = -np.sin(psi)
    m[1, 0] = np.sin(psi)
    m[1, 1] = np.cos(psi)
    m[2, 2] = 1
    return m

def Ry(phi):
    m = np.zeros([3, 3])
    m[0, 0] = np.cos(phi)
    m[0, 2] = np.sin(phi)
    m[1, 1] = 1
    m[2, 0] = -np.sin(phi)
    m[2, 2] = np.cos(phi)
    return m

def Rx(theta):
    m = np.zeros([3, 3])
    m[0, 0] = 1
    m[1, 1] = np.cos(theta)
    m[1, 2] = -np.sin(theta)
    m[2, 1] = np.sin(theta)
    m[2, 2] = np.cos(theta)
    return m

def pc2obj(filepath,pc):
    nverts = pc.shape[1]
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in range(nverts):
            f.write("v %.4f %.4f %.4f\n" % (pc[0,v],pc[1,v],pc[2,v]))

def render_plane_point(R_src, R_tgt):
    r = 2.3
    origin = [0, 0, 0]
    dx = [-1, 1, 1, -1, -1, 1, 1, -1]
    dy = [-1, -1, 1, 1, -1, -1, 1, 1]
    dz = [-1, -1, -1, -1, 1, 1, 1, 1]
    pts3d_pl = []
    ptt3d_pl = []
    ptsns_pl = []
    pttnt_pl = []
    for i in range(8):
        x = origin[0] + r * dx[i]
        y = origin[0] + r * dy[i]
        z = origin[0] + r * dz[i]
        point_ = [x, y, z]
        pts3d_pl.append(point_)
        ptt3d_pl.append(point_)
        normal_ = point_ / np.sqrt(np.power(point_, 2).sum())
        ptsns_pl.append(normal_)
        pttnt_pl.append(normal_)
    

    pts3d_pl = np.matmul(R_src, np.asarray(pts3d_pl).T)
    ptt3d_pl = np.matmul(R_tgt, np.asarray(ptt3d_pl).T)
    ptsns_pl = np.matmul(R_src, np.asarray(ptsns_pl).T)
    pttnt_pl = np.matmul(R_tgt, np.asarray(pttnt_pl).T)

    return pts3d_pl, ptt3d_pl, ptsns_pl, pttnt_pl

def render_random_plane_point():
    pts3d_pl = np.random.rand(8,3)
    ptt3d_pl = np.random.rand(8,3)

    p_src_mean = pts3d_pl.sum(0) / 8.
    p_tgt_mean = ptt3d_pl.sum(0) / 8.

    p_src = pts3d_pl - p_src_mean[np.newaxis,:]
    p_tgt = ptt3d_pl - p_tgt_mean[np.newaxis,:]

    p_src_n = []
    p_tgt_n = []
    for i in range(8):
        normal_s = p_src[i] / np.sqrt(np.power(p_src[i],2).sum())
        normal_t = p_tgt[i] / np.sqrt(np.power(p_tgt[i],2).sum())
        p_src_n.append(normal_s[np.newaxis,:])
        p_tgt_n.append(normal_t[np.newaxis,:])

    p_src_n = np.concatenate(p_src_n, axis=0)
    p_tgt_n = np.concatenate(p_tgt_n, axis=0)


    return pts3d_pl, ptt3d_pl, p_src_n, p_tgt_n

def process_corner_point(src_path, tar_path):
    partition_src = src_path[35:].strip(' ')
    partition_tar = tar_path[35:].strip(' ')

    src_plane = np.load('./data/test_data/suncg100/' + partition_src.replace("/", "-") + '.npy')
    tar_plane = np.load('./data/test_data/suncg100/' + partition_tar.replace("/", "-") + '.npy')
    gt_src = src_plane.item()['corner_gt']
    pred_src = src_plane.item()['corner_pred']
    gt_tar = tar_plane.item()['corner_gt']
    pred_tar = tar_plane.item()['corner_pred']
    
    return gt_src.T, pred_src.T, gt_tar.T, pred_tar.T

def process_plane_point(src_path, tar_path, dataList): 
    if isinstance(src_path, np.ndarray):
        src_path = src_path[0]
        tar_path = tar_path[0]

    partition_src = src_path.strip('/').strip(' ')
    partition_tar = tar_path.strip('/').strip(' ')

    if 'suncg' in dataList:
        # suncg data path
        basepath = './data/eval/plane_data/suncg/'  
    elif 'scannet' in dataList:
        # scannet data path

        basepath = './data/eval/plane_data/scannet/'
        partition_src = partition_src.split('ScanNet_360/test/')[-1]
        partition_tar = partition_tar.split('ScanNet_360/test/')[-1]

    
    elif 'matterport' in dataList:
        basepath = './data/eval/plane_data/matterport/'
        partition_src = '/'.join(partition_src.split('/')[-2:])
        partition_tar = '/'.join(partition_tar.split('/')[-2:])

    src_path = basepath + partition_src.replace("/", "-") + '.npy'
    tgt_path = basepath + partition_tar.replace("/", "-") + '.npy'

    if os.path.exists(src_path) and os.path.exists(tgt_path):

        src_plane = np.load(basepath + partition_src.replace("/", "-") + '.npy', allow_pickle=True)
        tar_plane = np.load(basepath + partition_tar.replace("/", "-") + '.npy', allow_pickle=True)
        
        if 'matterport' in dataList:
            gt_src = 0
            gt_tar = 0
        else:
            gt_src = src_plane.item()['plane_gt']
            gt_tar = tar_plane.item()['plane_gt']

        pred_src = src_plane.item()['plane_pred']
        pred_tar = tar_plane.item()['plane_pred']  

        return gt_src, pred_src, gt_tar, pred_tar
    else:

        return 0,0,0,0

def process_plane_pointV2(src_path, tar_path, dataList): 
    if isinstance(src_path, np.ndarray):
        src_path = src_path[0]
        tar_path = tar_path[0]

    partition_src = src_path.strip('/').strip(' ')
    partition_tar = tar_path.strip('/').strip(' ')
    if 'suncg' in dataList:
        # suncg data path
        basepath = './data/eval/plane_data/suncg/' 
        basepath2 = './data/eval/plane_data/suncg_obj/'
    elif 'scannet' in dataList:
        basepath = './data/eval/plane_data/scannet/'
        partition_src = partition_src.split('ScanNet_360/test/')[-1]
        partition_tar = partition_tar.split('ScanNet_360/test/')[-1]
        basepath2 = './data/eval/plane_data/scannet_obj/'
    elif 'matterport' in dataList:
        basepath = './data/eval/plane_data/matterport/'
        partition_src = '/'.join(partition_src.split('/')[-2:])
        partition_tar = '/'.join(partition_tar.split('/')[-2:])

    src_path = basepath + partition_src.replace("/", "-") + '.npy'
    tgt_path = basepath + partition_tar.replace("/", "-") + '.npy'

    src_path2 = basepath2 + partition_src.replace("/", "-") + '.npy'
    tgt_path2 = basepath2 + partition_tar.replace("/", "-") + '.npy'
    
    if os.path.exists(src_path) and os.path.exists(tgt_path) and os.path.exists(src_path2) and os.path.exists(tgt_path2):

        src_plane = np.load(basepath + partition_src.replace("/", "-") + '.npy', allow_pickle=True)
        tar_plane = np.load(basepath + partition_tar.replace("/", "-") + '.npy', allow_pickle=True)
        
        src_plane2 = np.load(basepath2 + partition_src.replace("/", "-") + '.npy', allow_pickle=True)
        tar_plane2 = np.load(basepath2 + partition_tar.replace("/", "-") + '.npy', allow_pickle=True)       
        
        gt_src = 0
        gt_tar = 0

        pred_src = src_plane.item()['plane_pred']
        pred_tar = tar_plane.item()['plane_pred']  

        pred_src2 = src_plane2.item()['plane_pred'][:,:6]
        pred_tar2 = tar_plane2.item()['plane_pred'][:,:6]

        pred_src = np.concatenate([pred_src, pred_src2])
        pred_tar = np.concatenate([pred_tar, pred_tar2])

        return gt_src, pred_src, gt_tar, pred_tar
    else:

        return 0,0,0,0


def process_topdown_mat(src_path, tgt_path, dataList):
    
    if isinstance(src_path, np.ndarray):
        src_path = src_path[0]
        tgt_path = tgt_path[0]

    clean_src = src_path.strip('/').strip(' ')
    clean_tgt = tgt_path.strip('/').strip(' ')
    partition_src_tgt = clean_src.split('/')[-3:]
    if 'scannet' in dataList:

        data_path_s = './data/eval/topdown_data/scannet/' + '-'.join(clean_src.split('/')[-2:]) + '.npy'
        data_path_t = './data/eval/topdown_data/scannet/' + '-'.join(clean_tgt.split('/')[-2:]) + '.npy'

        if os.path.exists(data_path_s) and os.path.exists(data_path_t):

            data_s = np.load(data_path_s, allow_pickle=True).item()
            data_t = np.load(data_path_t, allow_pickle=True).item()
            topdown_data = {}
            topdown_data['feat_s'] = data_s['feat']
            topdown_data['feat_t'] = data_t['feat']
            topdown_data['nor_s'] = data_s['nor']
            topdown_data['nor_t'] = data_t['nor']
            topdown_data['pos_s'] = data_s['pos']
            topdown_data['pos_t'] = data_t['pos']
            return topdown_data
        else:
            return 0
    elif 'matterport' in dataList:
        #import pdb; pdb.set_trace()
        partition_src_tgt.append(clean_tgt.split('/')[-1])

        data_path = './data/eval/topdown_data/matterport/' + "-".join(partition_src_tgt) + '.npy'

        if os.path.exists(data_path):
            #import pdb; pdb.set_trace()
            topdown_data = np.load(data_path, allow_pickle=True).item()
            return topdown_data
        else:
            return 0

    elif 'suncg' in dataList:
        partition_src_tgt.append(clean_tgt.split('/')[-1])
        data_path = './data/eval/topdown_data/suncg/' + "-".join(partition_src_tgt) + '.npy'

        if os.path.exists(data_path):
            #import pdb; pdb.set_trace()
            topdown_data = np.load(data_path, allow_pickle=True).item()
            return topdown_data
        else:
            return 0

def bilinear_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    return wa[:,None]*Ia + wb[:,None]*Ib + wc[:,None]*Ic + wd[:,None]*Id
def nearest_interpolate(im, x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    # if x out of range, assign zero 
    mask = (x < 0) | (x >= im.shape[1] - 1)
    mask = mask | ((y < 0) | (y >= im.shape[1] - 1))
    
    x0 = np.clip(x0, 0, im.shape[1]-1)
    x1 = np.clip(x1, 0, im.shape[1]-1)
    y0 = np.clip(y0, 0, im.shape[0]-1)
    y1 = np.clip(y1, 0, im.shape[0]-1)

    Ia = im[ y0, x0 ]
    Ib = im[ y1, x0 ]
    Ic = im[ y0, x1 ]
    Id = im[ y1, x1 ]

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)
    
    idx = np.argmax(np.stack((wa,wb,wc,wb),1),1)
    I = np.stack((Ia,Ib,Ic,Id),1)
    
    I[mask] = 0
    return I[range(idx.shape[0]), idx]
def topdown_projection(pc, pc_s, colors, origin, axis_x, axis_y, axis_z, height=400, width=400, resolution=0.02):
    u = ((pc - origin[None,:]) * axis_x[None,:]).sum(1)
    v = ((pc - origin[None,:]) * axis_y[None,:]).sum(1)
    z = ((pc - origin[None,:]) * axis_z[None,:]).sum(1)
    # write_ply('test.ply',np.stack((u,v,z),-1), color=colors[pc_s])

    u = width//2 + (u / resolution).astype('int')
    v = height//2 - (v / resolution).astype('int')
    
    pc_s_color = colors[pc_s]
    topdown_c = np.zeros([height, width, 3])
    topdown_s = np.zeros([height, width])
    topdown_h = np.zeros([height, width])
    
    disk = 3
    for i in range(len(pc)):
        if u[i] < 0 | u[i] > width-1 | v[i] < 0 | v[i] > height-1:
            continue 
        else:
            mask = (topdown_h[(v[i]-disk).clip(0, height-1):(v[i]+disk+1).clip(0, height-1), (u[i]-disk).clip(0, width-1):(u[i]+disk+1).clip(0, width-1)] < z[i])[:,:,None]
            topdown_c[(v[i]-disk).clip(0, height-1):(v[i]+disk+1).clip(0, height-1), (u[i]-disk).clip(0, width-1):(u[i]+disk+1).clip(0, width-1)][mask[:,:,0]] = pc_s_color[i]
            topdown_s[(v[i]-disk).clip(0, height-1):(v[i]+disk+1).clip(0, height-1), (u[i]-disk).clip(0, width-1):(u[i]+disk+1).clip(0, width-1)][mask[:,:,0]] = pc_s[i]
            topdown_h[(v[i]-disk).clip(0, height-1):(v[i]+disk+1).clip(0, height-1), (u[i]-disk).clip(0, width-1):(u[i]+disk+1).clip(0, width-1)][mask[:,:,0]] = z[i]
    
    #mask = (topdown_c==0).sum(2)==3
    #topdown_c[mask]=1
    
    # quantize z axis into 3 bins
    ind_z = np.digitize(z, [-0.1, 0.7, 1.5])
    ind = np.stack((u, v, ind_z), -1)
    return topdown_c, topdown_s, ind

def normal_plane_point(p_src, p_tgt):
    '''
    p_src: [n,3]
    p_tgt: [n,3]
    '''
    p_src_mean = p_src.sum(0) / p_src.shape[0]
    p_tgt_mean = p_tgt.sum(0) / p_tgt.shape[0]
    

    p_src -= p_src_mean[np.newaxis,:]
    p_tgt -= p_tgt_mean[np.newaxis,:]

    p_src_n = []
    p_tgt_n = []
    for i in range(p_src.shape[0]):
        normal_s = p_src[i] / np.sqrt(np.power(p_src[i],2).sum())
        normal_t = p_tgt[i] / np.sqrt(np.power(p_tgt[i],2).sum())
        p_src_n.append(normal_s[np.newaxis,:])
        p_tgt_n.append(normal_t[np.newaxis,:])

    p_src_n = np.concatenate(p_src_n, axis=0)
    p_tgt_n = np.concatenate(p_tgt_n, axis=0)

    return p_src_n, p_tgt_n

def plane_eq2point(plane_eq):
    # plane_eq : [b, n, 4]
    normal_ = plane_eq[:,:,:3]
    plane_dis = plane_eq[:,:,3]

    loc = normal_ * plane_dis.unsqueeze(2)
 
    return loc

def get_plane_verts(plane_center, a, b, c, d):


    color = np.random.rand(3)
    xx, yy = np.mgrid[-3:3:0.1, -3:3:0.1]
    xx = xx.flatten()
    yy = yy.flatten()
    xx += plane_center[0]
    yy += plane_center[1]

    zz = (-d - a * xx - b * yy) / c
    vert_plane = np.stack((xx, yy, zz),-1)
    color_plane = np.tile(color[None, :], [len(xx), 1])

    return vert_plane, color_plane

def point2plane_eq(points):
    d = np.sqrt(np.sum(np.power(points,2)))
    a, b, c = points / d
    return [a, b, c, d]

def ComputeBasis(nor):
    signN = 1 if nor[2] >0 else -1
    a = -1/(signN + nor[2])
    b = nor[0] * nor[1] * a
    b1 = np.array([1 + signN*nor[0]*nor[0]*a,signN*b,-signN*nor[0]])
    b2 = np.array([b, signN + nor[1]*nor[1]*a, -nor[1]])
    return b1, b2 

def draw_wall_bbox(centers):
    center = np.mean(centers,axis=0)
    tmp_1 = centers[-1]
    c_tmp_1 = 2*center - centers[-1]

    for i in range(6):
        if centers[i].sum() != tmp_1.sum() and np.sqrt(np.power(centers[i] - c_tmp_1,2).sum()) >1:
            tmp_2 = centers[i]
    
    tmp_1 = tmp_1 - center
    tmp_1 = tmp_1 / np.sqrt(np.power(tmp_1,2).sum())
    tmp_2 = tmp_2 - center
    tmp_2 = tmp_2 / np.sqrt(np.power(tmp_2,2).sum())
    tmp_3 = np.cross(tmp_1,tmp_2)

    trans_matrix = np.concatenate([tmp_1[None,:], tmp_2[None,:], tmp_3[None,:]]).T

    new_centers = np.matmul(np.linalg.inv(trans_matrix), centers.T).T

    xmin = np.min(new_centers[:,0])
    xmax = np.max(new_centers[:,0])
    ymin = np.min(new_centers[:,1])
    ymax = np.max(new_centers[:,1])
    zmin = np.min(new_centers[:,2])
    zmax = np.max(new_centers[:,2])
    
    pt_1 = np.asarray([xmax,ymin,zmin])
    pt_2 = np.asarray([xmax,ymax,zmin])
    pt_3 = np.asarray([xmin,ymax,zmin])
    pt_4 = np.asarray([xmin,ymin,zmin])
    pt_5 = np.asarray([xmax,ymin,zmax])
    pt_6 = np.asarray([xmax,ymax,zmax])
    pt_7 = np.asarray([xmin,ymax,zmax])
    pt_8 = np.asarray([xmin,ymin,zmax])
    
    panel_1 = np.concatenate([pt_1[None,:], pt_2[None,:], pt_4[None,:], pt_3[None,:]])
    panel_2 = np.concatenate([pt_1[None,:], pt_2[None,:], pt_5[None,:], pt_6[None,:]])
    panel_3 = np.concatenate([pt_4[None,:], pt_1[None,:], pt_8[None,:], pt_5[None,:]])
    panel_4 = np.concatenate([pt_3[None,:], pt_2[None,:], pt_7[None,:], pt_6[None,:]])
    panel_5 = np.concatenate([pt_5[None,:], pt_6[None,:], pt_8[None,:], pt_7[None,:]])
    panel_6 = np.concatenate([pt_4[None,:], pt_3[None,:], pt_8[None,:], pt_7[None,:]])
    
    bbox_list = []
    color_bbox_list = []
    panel_list = [panel_1, panel_2, panel_3, panel_4, panel_5, panel_6]
    for panel in panel_list:
        back_panel = np.matmul(trans_matrix,panel.T).T
        bbox_tmp,color_bbox_tmp = draw_plane_bbox(back_panel)
        bbox_list.append(bbox_tmp)
        color_bbox_list.append(color_bbox_tmp)

    return bbox_list, color_bbox_list

def draw_plane_bbox(bbox):
    line_1,color = pcloud_line(bbox[0], bbox[1])
    line_2,color = pcloud_line(bbox[0], bbox[2])
    line_3,color = pcloud_line(bbox[1], bbox[3])
    line_4,color = pcloud_line(bbox[2], bbox[3])

    lines = np.concatenate([line_1, line_2, line_3, line_4])
    colors= np.concatenate([color,color,color,color])
    
    return lines, colors

def LoadImage(PATH,depth=True):
    
    if depth:
        img = cv2.imread(PATH,2) / 1000.
    else:
        img = cv2.imread(PATH)

    return img




def filter_corres(pt_s, pt_t, R_gt_44):
    #if pt_s.shape[0] < 10:
    #    return pt_s, pt_t
    #import pdb; pdb.set_trace()
    pt_s_tmp = pt_s
    pt_t_tmp = pt_t
    if 1:
        pt_s2t = (np.matmul(R_gt_44[:3,:3], pt_s.T) + R_gt_44[:3,3:4]).T

        dis = np.sqrt(np.power(pt_s2t - pt_t,2).sum(1))

        pt_s_tmp = pt_s[dis < 1,:]
        pt_t_tmp = pt_t[dis < 1,:]
        
        if pt_s_tmp.shape[0] == 0:
            return pt_s, pt_t

        if pt_s_tmp.shape[0] < 10:
            return pt_s_tmp, pt_t_tmp

    inx = np.random.choice(pt_s_tmp.shape[0], 15)

    pt_s = pt_s_tmp[inx,:]
    pt_t = pt_t_tmp[inx,:]
    return pt_s, pt_t

def draw_full_plane_bbox(full_pc, full_pc_n, partial_pc):
    import plane_utils
    
    plane_params_room, plane_idx_room = plane_utils.fit_planes(full_pc)
    colors = np.random.rand(200,3)

    plane_params, plane_idx_ = plane_utils.fit_planes(partial_pc)
    pc_s = partial_pc

    planeidconverter = {}
    for j in range(1, int(plane_idx_.max()+1)):
        belongtothis = pc_s[(plane_idx_ == j)]
        dst_min = 1000

        for k in range(len(plane_params_room)):
            dst = np.abs((belongtothis * plane_params_room[k][None,:3]).sum(1) + plane_params_room[k][3]).mean()
            print(dst)
            if dst < dst_min:
                dst_min = dst 
                planeidconverter[j] = (k +1, dst_min)
    newplaneid = np.zeros([len(plane_idx_)])
    for key in planeidconverter:
        newplaneid[plane_idx_==key] = planeidconverter[key][0]
    newplaneid_set = set(newplaneid)
    nmax = len(plane_params_room)
    largeplane = []
    for x in range(5):
        largeplane.append(nmax-x)

    v_all = []
    v_c_all = []
    vv_centers = []
    vv_normals = []
    for i in range(len(plane_params_room)):
        #if (i+1) not in newplaneid_set and (i+1) not in largeplane:
        if (i+1) not in newplaneid_set:
            continue
        #if i < 15: continue
        pc_p = full_pc[plane_idx_room == i+1]
        p_normal = full_pc_n[plane_idx_room == i+1]
        #import pdb; pdb.set_trace()
        #p_normal = np.mean(p_normal,0)
        p_center = np.mean(pc_p, 0) + np.random.randn(3)*1e-2
        p_n = plane_params_room[i][:3]
        b1, b2 = ComputeBasis(p_n)
        projx = ((pc_p - p_center) * b1[None, :]).sum(1)
        projy = ((pc_p - p_center) * b2[None, :]).sum(1)
        xmax = np.max(projx) + np.random.randn()*1e-2
        xmin = np.min(projx) + np.random.randn()*1e-2
        ymax = np.max(projy) + np.random.randn()*1e-2
        ymin = np.min(projy) + np.random.randn()*1e-2
        corners = [p_center + xmax * b1 + ymax * b2,
        p_center + xmax * b1 + ymin * b2,
        p_center + xmin * b1 + ymin * b2,
        p_center + xmin * b1 + ymax * b2,
        ]
        verts, verts_c = [], []
        vv_center, v_c_center, _ = pcloud_point(p_center, color=colors[i+1],eps=1e-2)
        vv_centers.append(p_center[None,:])
        vv_normals.append(p_n[None,:])
        #verts.append(vv)
        #verts_c.append(v_c)

        
        vv,v_c = pcloud_line(corners[0], corners[1], color=colors[i+1])
        
        verts.append(vv)
        verts_c.append(v_c)
        
        vv,v_c = pcloud_line(corners[1], corners[2], color=colors[i+1])
        verts.append(vv)
        verts_c.append(v_c)
        vv,v_c = pcloud_line(corners[2], corners[3], color=colors[i+1])
        verts.append(vv)
        verts_c.append(v_c)
        vv,v_c = pcloud_line(corners[3], corners[0], color=colors[i+1])
        verts.append(vv)
        verts_c.append(v_c)
        verts = np.concatenate(verts)
        verts_c = np.concatenate(verts_c)
        v_all.append(verts)
        v_c_all.append(verts_c)

    #v_all = np.concatenate(v_all)
    v_c_all = np.concatenate(v_c_all)
    vv_centers = np.concatenate(vv_centers)
    vv_normals = np.concatenate(vv_normals)

    return v_all

# 3b33d5b6677a12f23e99ffbaac744a91-room_0_0_000001-000014
def transpose_process(pts, room_center, dirs=None):

    room_center = room_center[None,:]

    tmp_pts = pts

    Rs = np.array([[1,0,0],[0,0,-1],[0,1,0]])
    #tmp_pts = np.matmul(Rs, tmp_pts.T).T
    tmp_pts[:,2] = 2*room_center[:,2] - pts[:,2]
    #tmp_pts[:,1] = 2*room_center[:,1] - pts[:,1]
    tmp_pts[:,0] = 2*room_center[:,0] - pts[:,0]
    
    if dirs is not None:
        import pdb; pdb.set_trace()
        tmp_pts = tmp_pts + dirs[None,:]*-1
    else:
        tmp_pts = tmp_pts + np.asarray([[-14,5,-1]])

    return tmp_pts
    #return pts

def draw_correspondence(sourcePC, targetPC, corres, path_s, path_t, R_s, R_t, points_num, points_tgt_num, room_center):
    
    R_gt_44 = np.matmul(R_t, np.linalg.inv(R_s))
    sourcePC = (np.matmul(R_s[:3,:3],sourcePC.T)+R_s[:3,3:4]).T
    targetPC = (np.matmul(R_t[:3,:3],targetPC.T)+R_t[:3,3:4]).T

    corres = corres.astype('int')

    base_s = '/'.join(path_s.split('/')[:-1])
    base_t = '/'.join(path_t.split('/')[:-1])

    scan_s = path_s.split('/')[-1].strip(' ')
    scan_t = path_t.split('/')[-1].strip(' ')
    #import pdb; pdb.set_trace()
    base_path = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/SkyBox/SUNCG/'
    base_s = base_path + base_s.split('SUNCG/')[-1]
    base_t = base_path + base_t.split('SUNCG/')[-1]          
    #import pdb; pdb.set_trace()
    depth_s = cv2.imread(os.path.join(base_s, 'depth', scan_s + '.png'), 2) / 1000.
    depth_t = cv2.imread(os.path.join(base_t, 'depth', scan_t + '.png'), 2) / 1000.

    normal_s = LoadImage(os.path.join(base_s, 'normal', scan_s + '.png'),depth=False)/255.
    normal_t = LoadImage(os.path.join(base_t, 'normal', scan_t + '.png'),depth=False)/255.
    normal_s = normal_s[:,160*2:]
    normal_t = normal_t[:,160*2:]
    normal_s = np.concatenate((normal_s[:,:160].reshape(-1,3),
    normal_s[:,160:2*160].reshape(-1,3),
    normal_s[:,2*160:3*160].reshape(-1,3),
    normal_s[:,3*160:].reshape(-1,3)))
    normal_t = np.concatenate((normal_t[:,:160].reshape(-1,3),
    normal_t[:,160:2*160].reshape(-1,3),
    normal_t[:,2*160:3*160].reshape(-1,3),
    normal_t[:,3*160:].reshape(-1,3)))

    # observed region   
    source_pt, mask_s = depth2pc(depth_s[:,160*3:160*4], 'suncg')
    target_pt, mask_t = depth2pc(depth_t[:,160*3:160*4], 'suncg')
    source_pt = (np.matmul(R_s[:3,:3],source_pt.T)+R_s[:3,3:4]).T
    target_pt = (np.matmul(R_t[:3,:3],target_pt.T)+R_t[:3,3:4]).T 
    
    # full region
    full_pc_s = Pano2PointCloud(depth_s[:,160*2:], 'suncg').T
    full_pc_t = Pano2PointCloud(depth_t[:,160*2:], 'suncg').T


    Rs_ = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    R_s_ = np.matmul(R_s, np.linalg.inv(Rs_))
    R_t_ = np.matmul(R_t, np.linalg.inv(Rs_))
    full_pc_s = (np.matmul(R_s_[:3,:3],full_pc_s.T)+R_s_[:3,3:4]).T
    full_pc_t = (np.matmul(R_t_[:3,:3],full_pc_t.T)+R_t_[:3,3:4]).T

    if 0:
        #import pdb; pdb.set_trace()
        pc_s_planes = draw_full_plane_bbox(full_pc_s, normal_s, source_pt)
        
        for i in range(len(pc_s_planes)):
            cur_planes = transpose_process(pc_s_planes[i], room_center)

            write_ply('../visualization/corres/%s_%s-%s_plane_bbox_s_%d.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t, i), cur_planes)
        

        pc_t_planes = draw_full_plane_bbox(full_pc_t, normal_t, target_pt)

        for i in range(len(pc_t_planes)):
            cur_planes = pc_t_planes[i]
            write_ply('../visualization/corres/%s_%s-%s_plane_bbox_t_%d.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t, i), cur_planes)   
   




    rgb_s = LoadImage(os.path.join(base_s, 'rgb', scan_s + '.png'),depth=False)/255.
    rgb_t = LoadImage(os.path.join(base_t, 'rgb', scan_t + '.png'),depth=False)/255. 
    #import pdb; pdb.set_trace()

    pc_s_c = rgb_s[:,160*3:160*4].reshape(-1,3)[mask_s]
    pc_t_c = rgb_t[:,160*3:160*4].reshape(-1,3)[mask_t]


    
    #import pdb; pdb.set_trace() 
    
    corres_360 = corres[:,corres[0] < points_num[1]]
    corres = corres[:,corres[0] >= points_num[1]]

    corres_plane = corres[:,corres[0] < points_num[2]]
    corres = corres[:,corres[0] >= points_num[2]]
    corres_topdown = corres[:,corres[0] < points_num[3]]


    source_360_list = sourcePC[corres_360[0],:]
    target_360_list = targetPC[corres_360[1],:]
    #import pdb; pdb.set_trace()
    source_360_list, target_360_list = filter_corres(source_360_list, target_360_list, R_gt_44)

    # draw 360 lines
    source_360_list = transpose_process(source_360_list, room_center) 

    lines = []
    color_lines = []
    for i in range(source_360_list.shape[0]):
        pc_lines, color_line = pcloud_line(source_360_list[i], target_360_list[i], color=np.array([255,0,0]))
        lines.append(pc_lines)
        color_lines.append(color_line)
    
    lines_360 = np.concatenate(lines)
    color_360 = np.concatenate(color_lines)
    
    #room_center = sourcePC[corres_plane[0],:].mean(0)
    
    source_plane_list = sourcePC[corres_plane[0],:]
    target_plane_list = targetPC[corres_plane[1],:]

    #source_plane_list, target_plane_list = filter_corres(source_plane_list, target_plane_list, R_gt_44)
    # draw plane lines
    source_plane_list = transpose_process(source_plane_list, room_center)

    lines = []
    color_lines = []
    for i in range(source_plane_list.shape[0]):
        pc_lines, color_line = pcloud_line(source_plane_list[i], target_plane_list[i], color=np.array([0,255,0]))
        write_ply('../visualization/corres/%s_%s-%s_plane_lines_%d.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t, i), pc_lines, color=color_line)
        lines.append(pc_lines)
        color_lines.append(color_line)
    
    lines_plane = np.concatenate(lines)
    color_plane = np.concatenate(color_lines)

    
    # topdown
    source_topdown_list = sourcePC[corres_topdown[0],:]
    target_topdown_list = targetPC[corres_topdown[1],:]

    source_topdown_list, target_topdown_list = filter_corres(source_topdown_list, target_topdown_list, R_gt_44)
    # draw topdown lines
    source_topdown_list = transpose_process(source_topdown_list, room_center)

    lines = []
    color_lines = []
    for i in range(source_topdown_list.shape[0]):
        pc_lines, color_line = pcloud_line(source_topdown_list[i], target_topdown_list[i], color=np.array([0,0,255]))
        lines.append(pc_lines)
        color_lines.append(color_line)
    
    lines_topdown = np.concatenate(lines)
    color_topdown = np.concatenate(color_lines)

    # write into files
    write_ply('../visualization/corres/%s_%s-%s_plane_lines.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), lines_plane, color=color_plane)
    
    write_ply('../visualization/corres/%s_%s-%s_360_lines.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), lines_360, color=color_360)

    write_ply('../visualization/corres/%s_%s-%s_topdown_lines.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), lines_topdown, color=color_topdown)



    #source_td = sourcePC[points_num[2]:,:]
    #target_td = targetPC[points_tgt_num[2]:,:]
    #import pdb; pdb.set_trace()
    source_td = np.load('./data/test_data/topdown_supp_data/%s_%s.npy' % ('-'.join(base_s.split('/')[-2:]), scan_s),allow_pickle=True).item()

    target_td = np.load('./data/test_data/topdown_supp_data/%s_%s.npy' % ('-'.join(base_s.split('/')[-2:]), scan_t),allow_pickle=True).item()
    
    td_s = source_td['pc']
    color_td_s = source_td['color']
    td_t = target_td['pc']
    color_td_t = target_td['color']

    td_s = (np.matmul(R_s[:3,:3],td_s.T)+R_s[:3,3:4]).T
    td_t = (np.matmul(R_t[:3,:3],td_t.T)+R_t[:3,3:4]).T
    
    td_s = transpose_process(td_s, room_center)

    write_ply('../visualization/corres/%s_%s-%s_test_topdown.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), np.concatenate([td_s, td_t]), color=np.concatenate([color_td_s,color_td_t])) 
        
    color_s = np.random.rand(3)
    color_t = np.random.rand(3)
    color_s = np.tile(color_s[None,:], (source_pt.shape[0],1))
    color_t = np.tile(color_t[None,:], (target_pt.shape[0],1))

    #import pdb; pdb.set_trace()    
    #source_pt = transpose_process(source_pt, room_center, source_pt.mean(0)-target_pt.mean(0))   
    source_pt = transpose_process(source_pt, room_center)
    full_pc_s = transpose_process(full_pc_s, room_center)

    write_ply('../visualization/corres/%s_%s-%s_full_pt.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), np.concatenate([full_pc_s,full_pc_t]),color=np.concatenate([normal_s,normal_t]))

    write_ply('../visualization/corres/%s_%s-%s_pt.ply' % ('-'.join(base_s.split('/')[-2:]), scan_s, scan_t), np.concatenate([source_pt,target_pt]),color=np.concatenate([pc_s_c,pc_t_c]))
    import pdb; pdb.set_trace()
   


    
    


