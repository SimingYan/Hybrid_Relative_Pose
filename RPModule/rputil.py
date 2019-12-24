
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append('../')
import util
from utils import torch_op
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from sklearn.neighbors import NearestNeighbors

class opts():
    def __init__(self,sigmaAngle1=0.523/2,sigmaAngle2=0.523/2,sigmaDist=0.08/2,sigmaFeat=0.01):
        self.distThre = 0.08
        self.distSepThre = 1.5*0.08
        self.angleThre = 45/180.*np.pi
        self.sigmaAngle1=sigmaAngle1
        self.sigmaAngle2=sigmaAngle2
        self.sigmaDist=sigmaDist
        self.sigmaFeat=sigmaFeat
        self.mu = 0.3
        self.topK = 2
        self.method = 'irls+sm'
        self.hybrid_method = '360'
        self.numMatches = 6
        self.w_nor = 0.5
        self.w_fea = 0.1
        self.max_dist = 0.4
        self.w_pair = {'360':1, 'plane':np.sqrt(1), 'topdown':np.sqrt(1)}
        self.w_plane = 2
        self.w_topdown = 1

def angular_distance_np(R_hat, R):
    # measure the angular distance between two rotation matrice
    # R1,R2: [n, 3, 3]
    if len(R_hat.shape)==2:
        R_hat=R_hat[np.newaxis,:]
    if len(R.shape)==2:
        R=R[np.newaxis,:]
    n = R.shape[0]
    trace_idx = [0, 4, 8]
    trace = np.matmul(R_hat, R.transpose(0,2,1)).reshape(n,-1)[:,trace_idx].sum(1)
    metric = np.arccos(((trace - 1)/2).clip(-1,1)) / np.pi * 180.0
    return metric

def visNorm(vis):
    for v in range(len(vis)):
        if (vis[v].max().item() - vis[v].min().item())!=0:
            vis[v] = (vis[v]-vis[v].min())/(vis[v].max()-vis[v].min())
    return vis

def interpolate(feat, pt):
    # feat: c,h,w
    # pt: K,2
    # return: c,k

    h,w = feat.shape[1], feat.shape[2]
    x = pt[:,0] * (w-1)
    y = pt[:,1] * (h-1)
    x0 = torch.floor(x)
    y0 = torch.floor(y)

    val=feat[:,y0.long(),x0.long()]*(x0+1-x)*(y0+1-y)+\
            feat[:,y0.long()+1,x0.long()]*(x0+1-x)*(y-y0)+\
            feat[:,y0.long(),x0.long()+1]*(x-x0)*(y0+1-y)+\
            feat[:,y0.long()+1,x0.long()+1]*(x-x0)*(y-y0)
    
    return val


def getPixel_helper(depth,xs,ys,val,dataset='suncg',representation='skybox'):
    
    if dataset == 'scannet' and (depth.shape[0] == 200 and depth.shape[1] == 400):
      pc = []
      W, H = 400,200
      for i in range(len(xs)):
        ystp, xstp = (0.5 - ys[i] / H)*2, (xs[i] / W-0.5)*2
        zstp = val[i]
        ystp, xstp = ystp * zstp, xstp * zstp
        hfov = 120.0
        vfov = 2*np.arctan(np.tan(hfov/2/180*np.pi)*200/400)/np.pi*180
        xstp=xstp * np.tan(np.deg2rad(hfov/2))
        ystp=ystp * np.tan(np.deg2rad(vfov/2))
        tmp = np.concatenate(([xstp], [ystp], [-zstp]))
        pc.append(tmp)
      pc = np.concatenate(pc).reshape(-1, 3)
    else:
      assert(representation == 'skybox')
      W, H = 160,160
      assert(depth.shape[0] == H and depth.shape[1]== H*4 )
      
      #convert four views to the first view coordinate
      Rs = np.zeros([4, 4, 4])
      Rs[0] = np.eye(4)
      Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
      Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
      Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
      
      pc = []
      for i in range(len(xs)):
          idx = int(xs[i] // H)
          #if 'suncg' in dataset:
          #    R_this = Rs[idx]
          #elif 'scannet' in dataset or 'matterport' in dataset:
          #    R_this = Rs[(idx-1)%4]
          R_this = Rs[(idx-1)%4]
          ystp, xstp = (0.5 - ys[i] / H)*2, ((xs[i]- idx * H) / W-0.5)*2
          zstp = val[i]
          ystp, xstp = ystp * zstp, xstp * zstp
          tmp = np.concatenate(([xstp], [ystp], [-zstp]))
          tmp = np.matmul(R_this[:3, :3],tmp) + R_this[:3, 3]
          pc.append(tmp)
      pc = np.concatenate(pc).reshape(-1, 3)
    return pc

def getPixel(depth, normal, pts, dataset='suncg',representation='skybox'):
    # pts: [n, 2]
    # depth: [dim, dim]
    tp = np.floor(pts).astype('int')
    v1 = depth[tp[:,1],tp[:,0]]
    v2 = depth[tp[:,1],tp[:,0]+1]
    v3 = depth[tp[:,1]+1,tp[:,0]]
    v4 = depth[tp[:,1]+1,tp[:,0]+1]
    val = v1*(tp[:,1]+1-pts[:,1])*(tp[:,0]+1-pts[:,0]) + \
        v2*(pts[:,0]-tp[:,0])*(tp[:,1]+1-pts[:,1]) + \
        v3*(pts[:,1]-tp[:,1])*(tp[:,0]+1-pts[:,0]) + \
        v4*(pts[:,0]-tp[:,0])*(pts[:,1]-tp[:,1])
    
    Rs = np.zeros([4,4,4])
    Rs[0] = np.eye(4)
    Rs[1] = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    Rs[2] = np.array([[-1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
    Rs[3] = np.array([[0,0,1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]])
    v1 = normal[tp[:,1],tp[:,0],:]
    v2 = normal[tp[:,1],tp[:,0]+1,:]
    v3 = normal[tp[:,1]+1,tp[:,0],:]
    v4 = normal[tp[:,1]+1,tp[:,0]+1,:]
    nn = v1*(tp[:,1]+1-pts[:,1])[:,np.newaxis]*(tp[:,0]+1-pts[:,0])[:,np.newaxis] + \
        v2*(pts[:,0]-tp[:,0])[:,np.newaxis]*(tp[:,1]+1-pts[:,1])[:,np.newaxis] + \
        v3*(pts[:,1]-tp[:,1])[:,np.newaxis]*(tp[:,0]+1-pts[:,0])[:,np.newaxis] + \
        v4*(pts[:,0]-tp[:,0])[:,np.newaxis]*(pts[:,1]-tp[:,1])[:,np.newaxis]

    nn /= np.linalg.norm(nn,axis=1,keepdims=True)
    if 'suncg' in dataset:
        # transform normal from first view coordinate to second view coordinate
        nn = np.matmul(Rs[1][:3,:3].T, nn.T).T
        
    ys, xs = pts[:,1],pts[:,0]
    pc = getPixel_helper(depth,xs,ys,val,dataset,representation).T

    return pc,nn

def drawMatch(img0,img1,src,tgt,color='b'):
    if len(img0.shape)==2:
      img0=np.expand_dims(img0,2)
    if len(img1.shape)==2:
      img1=np.expand_dims(img1,2)
    h,w = img0.shape[0],img0.shape[1]
    img = np.zeros([2*h,w,3])
    img[:h,:,:] = img0
    img[h:,:,:] = img1
    n = len(src)
    if color == 'b':
        color=(255,0,0)
    else:
        color=(0,255,0)
    for i in range(n):
      cv2.circle(img, (int(src[i,0]), int(src[i,1])), 3,color,-1)
      cv2.circle(img, (int(tgt[i,0]), int(tgt[i,1])+h), 3,color,-1)
      cv2.line(img, (int(src[i,0]),int(src[i,1])),(int(tgt[i,0]),int(tgt[i,1])+h),color,1)
    return img

def getKeypoint(rs,rt,feats,featt):
    H, W         = 160, 640
    N_SIFT_MATCH = 30
    N_RANDOM     = 30
    MARKER       = 0.99
    SIFT_THRE    = 0.02
    TOPK         = 2

    grays = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    grayt = cv2.cvtColor(rt, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = SIFT_THRE)
    
    grays = grays[:, H:H*2]
    (kps, _) = sift.detectAndCompute(grays, None)
    if not len(kps):
        return None,None,None,None,None,None
    pts = np.zeros([len(kps),2])
    for j,m in enumerate(kps):
        pts[j, :] = m.pt
    pts[:, 0] += H

    grayt = grayt[:, H:H*2]
    (kpt, _) = sift.detectAndCompute(grayt, None)
    if not len(kpt):
        return None,None,None,None,None,None
    ptt = np.zeros([len(kpt),2])
    for j,m in enumerate(kpt):
        ptt[j, :] = m.pt
    ptt[:, 0] += H

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:, 0] /= W
    ptsNorm[:, 1] /= H
    pttNorm = ptt.copy().astype('float')
    pttNorm[:, 0] /= W
    pttNorm[:, 1] /= H

    fs0 = interpolate(feats, torch_op.v(ptsNorm))
    ft0 = interpolate(featt, torch_op.v(pttNorm))

    # find the most probable correspondence using feature map
    C = feats.shape[0]
    fsselect = np.random.choice(range(pts.shape[0]), min(N_SIFT_MATCH, pts.shape[0]))
    ftselect = np.random.choice(range(ptt.shape[0]), min(N_SIFT_MATCH, ptt.shape[0]))
    dist = (fs0[:, fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)
    
    pttAug = Sampling(torch_op.npy(dist), TOPK)
    dist = (ft0[:, ftselect].unsqueeze(2) - feats.view(C, 1, -1)).pow(2).sum(0).view(len(ftselect), H, W)
    ptsAug = Sampling(torch_op.npy(dist), TOPK)

    pttAug = pttAug.reshape(-1, 2)
    ptsAug = ptsAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W-1) * (pttAug[:, 1] < H-1)
    pttAug = pttAug[valid]
    valid = (ptsAug[:, 0] < W-1) * (ptsAug[:, 1] < H-1)
    ptsAug = ptsAug[valid]

    pts = np.concatenate((pts, ptsAug))
    ptt = np.concatenate((ptt, pttAug))

    xs = (np.random.rand(N_RANDOM) * W).astype('int').clip(0, W-2)
    ys = (np.random.rand(N_RANDOM) * H).astype('int').clip(0, H-2)
    ptsrnd = np.stack((xs, ys), 1)
    valid = ((ptsrnd[:, 0] >= H) * (ptsrnd[:, 0] <= H*2))
    ptsrnd = ptsrnd[~valid]
    ptsrndNorm = ptsrnd.copy().astype('float')
    ptsrndNorm[:, 0] /= W
    ptsrndNorm[:, 1] /= H
    fs0 = interpolate(feats, torch_op.v(ptsrndNorm))
    fsselect = np.random.choice(range(ptsrnd.shape[0]), min(N_RANDOM, ptsrnd.shape[0]))
    dist = (fs0[:,fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)
    
    pttAug = Sampling(torch_op.npy(dist), TOPK)
    pttAug = pttAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W-1) * (pttAug[:, 1] < H-1)
    pttAug = pttAug[valid]
    pts = np.concatenate((pts, ptsrnd[fsselect]))
    ptt = np.concatenate((ptt, pttAug))

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:, 0] /= W
    ptsNorm[:, 1] /= H

    pttNorm = ptt.copy().astype('float')
    pttNorm[:, 0] /= W
    pttNorm[:, 1] /= H

    valid = (pts[:, 0] >= H) * (pts[:, 0] <= H*2)
    ptsW = np.ones(len(valid))
    ptsW[~valid] *= MARKER

    valid = (ptt[:, 0] >= H) * (ptt[:, 0] <= H*2)
    pttW = np.ones(len(valid))
    pttW[~valid] *= MARKER

    return pts,ptsNorm,ptsW,ptt,pttNorm,pttW

def getKeypoint_kinect_120fov(rs,rt,feats,featt,rs_full,rt_full):
    H, W         = 200, 400
    KINECT_W     = 640
    KINECT_H     = 480
    KINECT_FOV_W = 128
    KINECT_FOV_H = 96
    N_SIFT       = 300
    N_SIFT_MATCH = 30
    N_RANDOM     = 100
    MARKER       = 0.99
    SIFT_THRE    = 0.02
    TOPK         = 2
    
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = SIFT_THRE)
    grays = cv2.cvtColor(rs_full, cv2.COLOR_BGR2GRAY)
    (kps, _) = sift.detectAndCompute(grays, None)
    if not len(kps):
        return None,None,None,None,None,None
    pts = np.zeros([len(kps), 2])
    for j, m in enumerate(kps):
        pts[j, :] = m.pt
    pts[:,0] = pts[:,0] / KINECT_W * KINECT_FOV_W # the observed region size of kinect camera is [88x66]
    pts[:,1] = pts[:,1] / KINECT_H * KINECT_FOV_H
    pts[:,0] += W // 2 - KINECT_FOV_W // 2
    pts[:,1] += H // 2 - KINECT_FOV_H // 2

    grayt = cv2.cvtColor(rt_full, cv2.COLOR_BGR2GRAY)
    (kpt, _) = sift.detectAndCompute(grayt, None)
    
    #cv2.imwrite('test_t.png',cv2.drawKeypoints(grayt,kpt,outImage=np.array([]), color=(0,255,0), flags=0))
    #cv2.imwrite('test_s.png',cv2.drawKeypoints(grays,kps,outImage=np.array([]),color=(0,255,0), flags=0))
    if not len(kpt):
        return None,None,None,None,None,None
    ptt = np.zeros([len(kpt), 2])
    for j, m in enumerate(kpt):
        ptt[j, :] = m.pt
    ptt[:, 0] = ptt[:, 0] / KINECT_W * KINECT_FOV_W
    ptt[:, 1] = ptt[:, 1] / KINECT_H * KINECT_FOV_H
    ptt[:, 0] += W // 2 - KINECT_FOV_W // 2
    ptt[:, 1] += H // 2 - KINECT_FOV_H // 2

    pts = pts[np.random.choice(range(len(pts)), N_SIFT), :]
    ptt = ptt[np.random.choice(range(len(ptt)), N_SIFT), :]
    
    ptsNorm = pts.copy().astype('float')
    ptsNorm[:, 0] /= W
    ptsNorm[:, 1] /= H
    pttNorm = ptt.copy().astype('float')
    pttNorm[:, 0] /= W
    pttNorm[:, 1] /= H

    fs0 = interpolate(feats, torch_op.v(ptsNorm))
    ft0 = interpolate(featt, torch_op.v(pttNorm))
    
    # find the most probable correspondence using feature map
    C = feats.shape[0]
    fsselect = np.random.choice(range(pts.shape[0]), min(N_SIFT_MATCH, pts.shape[0]))
    ftselect = np.random.choice(range(ptt.shape[0]), min(N_SIFT_MATCH, ptt.shape[0]))
    dist = (fs0[:, fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)
    pttAug = Sampling(torch_op.npy(dist), TOPK)
    dist = (ft0[:, ftselect].unsqueeze(2) - feats.view(C, 1, -1)).pow(2).sum(0).view(len(ftselect), H, W)
    ptsAug = Sampling(torch_op.npy(dist), TOPK)

    pttAug = pttAug.reshape(-1, 2)
    ptsAug = ptsAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W-1) * (pttAug[:, 1] < H-1)
    pttAug = pttAug[valid]
    valid = (ptsAug[:, 0] < W-1) * (ptsAug[:, 1] < H-1)
    ptsAug = ptsAug[valid]

    pts = np.concatenate((pts, ptsAug))
    ptt = np.concatenate((ptt, pttAug))

    N=120
    xs = (np.random.rand(N) * W).astype('int').clip(0, W-2)
    ys = (np.random.rand(N) * H).astype('int').clip(0, H-2)
    ptsrnd = np.stack((xs, ys),1)

    # filter out observed region
    valid=((ptsrnd[:, 0] >= W//2 - KINECT_FOV_W//2) *(ptsrnd[:, 0] <= W//2 + KINECT_FOV_W//2)*(ptsrnd[:, 1] >= H//2 - KINECT_FOV_H//2) * (ptsrnd[:, 1] <= H//2 + KINECT_FOV_H // 2))
    ptsrnd = ptsrnd[~valid]

    ptsrndNorm = ptsrnd.copy().astype('float')
    ptsrndNorm[:, 0] /= W
    ptsrndNorm[:, 1] /= H
    fs0 = interpolate(feats,torch_op.v(ptsrndNorm))
    fsselect = np.random.choice(range(ptsrnd.shape[0]), min(N_RANDOM, ptsrnd.shape[0]))
    dist = (fs0[:, fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)

    pttAug = Sampling(torch_op.npy(dist), TOPK)
    pttAug = pttAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W - 1) * (pttAug[:, 1] < H - 1)
    pttAug = pttAug[valid]
    pts = np.concatenate((pts, ptsrnd[fsselect]))
    ptt = np.concatenate((ptt, pttAug))

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0] /= W
    ptsNorm[:,1] /= H

    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0] /= W
    pttNorm[:,1] /= H

    # hacks to get the points belongs to kinect observed region. 
    valid = ((pts[:, 0] >= W // 2 - KINECT_FOV_W // 2) * (pts[:, 0] <= W // 2 + KINECT_FOV_W // 2) * (pts[:, 1] >= H // 2 - KINECT_FOV_H // 2) * (pts[:, 1] <= H // 2 + KINECT_FOV_H // 2))
    ptsW = np.ones(len(valid))
    ptsW[~valid] *= MARKER

    valid = ((ptt[:, 0] >= W // 2 - KINECT_FOV_W // 2) * (ptt[:, 0] <= W // 2 + KINECT_FOV_W // 2) * (ptt[:, 1] >= H // 2 - KINECT_FOV_H // 2) * (ptt[:, 1] <= H // 2 + KINECT_FOV_H // 2))
    pttW = np.ones(len(valid))
    pttW[~valid] *= MARKER

    return pts,ptsNorm,ptsW,ptt,pttNorm,pttW

def getKeypoint_kinect(rs,rt,feats,featt,rs_full,rt_full):
    H, W         = 160, 640
    KINECT_W     = 640
    KINECT_H     = 480
    KINECT_FOV_W = 88
    KINECT_FOV_H = 66
    N_SIFT       = 300
    N_SIFT_MATCH = 30
    N_RANDOM     = 100
    MARKER       = 0.99
    SIFT_THRE    = 0.02
    TOPK         = 2

    grays = cv2.cvtColor(rs, cv2.COLOR_BGR2GRAY)
    grayt = cv2.cvtColor(rt, cv2.COLOR_BGR2GRAY)

    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold = SIFT_THRE)
    grays = cv2.cvtColor(rs_full, cv2.COLOR_BGR2GRAY)
    (kps, _) = sift.detectAndCompute(grays, None)
    if not len(kps):
        return None,None,None,None,None,None
    pts = np.zeros([len(kps), 2])
    for j, m in enumerate(kps):
        pts[j, :] = m.pt
    pts[:,0] = pts[:,0] / KINECT_W * KINECT_FOV_W # the observed region size of kinect camera is [88x66]
    pts[:,1] = pts[:,1] / KINECT_H * KINECT_FOV_H
    pts[:,0] += H + H // 2 - KINECT_FOV_W // 2
    pts[:,1] += H // 2 - KINECT_FOV_H // 2

    grayt = cv2.cvtColor(rt_full, cv2.COLOR_BGR2GRAY)
    (kpt, _) = sift.detectAndCompute(grayt, None)
    if not len(kpt):
        return None,None,None,None,None,None
    ptt = np.zeros([len(kpt), 2])
    for j, m in enumerate(kpt):
        ptt[j, :] = m.pt
    ptt[:, 0] = ptt[:, 0] / KINECT_W * KINECT_FOV_W
    ptt[:, 1] = ptt[:, 1] / KINECT_H * KINECT_FOV_H
    ptt[:, 0] += H + H // 2 - KINECT_FOV_W // 2
    ptt[:, 1] += H // 2 - KINECT_FOV_H // 2

    pts = pts[np.random.choice(range(len(pts)), N_SIFT), :]
    ptt = ptt[np.random.choice(range(len(ptt)), N_SIFT), :]
    
    ptsNorm = pts.copy().astype('float')
    ptsNorm[:, 0] /= W
    ptsNorm[:, 1] /= H
    pttNorm = ptt.copy().astype('float')
    pttNorm[:, 0] /= W
    pttNorm[:, 1] /= H

    fs0 = interpolate(feats, torch_op.v(ptsNorm))
    ft0 = interpolate(featt, torch_op.v(pttNorm))

    # find the most probable correspondence using feature map
    C = feats.shape[0]
    fsselect = np.random.choice(range(pts.shape[0]), min(N_SIFT_MATCH, pts.shape[0]))
    ftselect = np.random.choice(range(ptt.shape[0]), min(N_SIFT_MATCH, ptt.shape[0]))
    dist = (fs0[:, fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)
    pttAug = Sampling(torch_op.npy(dist), TOPK)
    dist = (ft0[:, ftselect].unsqueeze(2) - feats.view(C, 1, -1)).pow(2).sum(0).view(len(ftselect), H, W)
    ptsAug = Sampling(torch_op.npy(dist), TOPK)

    pttAug = pttAug.reshape(-1, 2)
    ptsAug = ptsAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W-1) * (pttAug[:, 1] < H-1)
    pttAug = pttAug[valid]
    valid = (ptsAug[:, 0] < W-1) * (ptsAug[:, 1] < H-1)
    ptsAug = ptsAug[valid]

    pts = np.concatenate((pts, ptsAug))
    ptt = np.concatenate((ptt, pttAug))

    N=120
    xs = (np.random.rand(N) * W).astype('int').clip(0, W-2)
    ys = (np.random.rand(N) * H).astype('int').clip(0, H-2)
    ptsrnd = np.stack((xs, ys),1)

    # filter out observed region
    valid=((ptsrnd[:, 0] >= H + H//2 - KINECT_FOV_W//2) *(ptsrnd[:, 0] <= H + H//2 + KINECT_FOV_W//2)*(ptsrnd[:, 1] >= H//2 - KINECT_FOV_H//2) * (ptsrnd[:, 1] <= H//2 + KINECT_FOV_H // 2))
    ptsrnd = ptsrnd[~valid]

    ptsrndNorm = ptsrnd.copy().astype('float')
    ptsrndNorm[:, 0] /= W
    ptsrndNorm[:, 1] /= H
    fs0 = interpolate(feats,torch_op.v(ptsrndNorm))
    fsselect = np.random.choice(range(ptsrnd.shape[0]), min(N_RANDOM, ptsrnd.shape[0]))
    dist = (fs0[:, fsselect].unsqueeze(2) - featt.view(C, 1, -1)).pow(2).sum(0).view(len(fsselect), H, W)

    pttAug = Sampling(torch_op.npy(dist), TOPK)
    pttAug = pttAug.reshape(-1, 2)
    valid = (pttAug[:, 0] < W - 1) * (pttAug[:, 1] < H - 1)
    pttAug = pttAug[valid]
    pts = np.concatenate((pts, ptsrnd[fsselect]))
    ptt = np.concatenate((ptt, pttAug))

    ptsNorm = pts.copy().astype('float')
    ptsNorm[:,0] /= W
    ptsNorm[:,1] /= H

    pttNorm = ptt.copy().astype('float')
    pttNorm[:,0] /= W
    pttNorm[:,1] /= H

    # hacks to get the points belongs to kinect observed region. 
    valid = ((pts[:, 0] >= H + H // 2 - KINECT_FOV_W // 2) * (pts[:, 0] <= H + H // 2 + KINECT_FOV_W // 2) * (pts[:, 1] >= H // 2 - KINECT_FOV_H // 2) * (pts[:, 1] <= H // 2 + KINECT_FOV_H // 2))
    ptsW = np.ones(len(valid))
    ptsW[~valid] *= MARKER

    valid = ((ptt[:, 0] >= H + H // 2 - KINECT_FOV_W // 2) * (ptt[:, 0] <= H + H // 2 + KINECT_FOV_W // 2) * (ptt[:, 1] >= H // 2 - KINECT_FOV_H // 2) * (ptt[:, 1] <= H // 2 + KINECT_FOV_H // 2))
    pttW = np.ones(len(valid))
    pttW[~valid] *= MARKER

    return pts,ptsNorm,ptsW,ptt,pttNorm,pttW

def Sampling(heatmap, K):
    # heatmap: [n,h,w]
    # return: [n,K,2]
    heatmap = np.exp(-heatmap/2)
    n,h,w=heatmap.shape
    pt = np.zeros([n,K,2])
    WINDOW_SZ = 15
    for i in range(n):
        for j in range(K):
            idx=np.argmax(heatmap[i])
            coord=np.unravel_index(idx,heatmap[i].shape)[::-1]
            pt[i,j,:]=coord
            # suppress the neighbors
            topl=[max(0,coord[0] - WINDOW_SZ),max(0,coord[1] - WINDOW_SZ)]
            botr=[min(w-1,coord[0] + WINDOW_SZ),min(h-1,coord[1] + WINDOW_SZ)]
            heatmap[i][topl[1]:botr[1],topl[0]:botr[0]] = heatmap[i].min()
    return pt

def greedy_rounding(matC, order):
    #import pdb; pdb.set_trace()
    # get the best correpondence pair based on the sequential of eigenvector's value
    # return the index of good correspondence pair
    #import pdb;pdb.set_trace()
    consCorrIds = []
    consCorrIds.append(order[0])
    #rows, cols = matC.nonzero()

    for i in range(1, len(order)):
        #source_ = (rows == order[i])
        #target_ = (cols == consCorrIds)

        s = np.sum(matC[order[i], consCorrIds] != 0)

        #x = np.sum(np.logical_and(source_, target_))
        ratio = s / len(consCorrIds)
        if ratio >= 0.75:
            consCorrIds.append(order[i])

    return consCorrIds


def consistency_matrix(sourcePC, targetPC, sourceNormal, targetNormal, candCorrs, corrs_num, hybrid_representation, para):
    
    sigmaDist = para.sigmaDist
    sigmaAngle1 = para.sigmaAngle1
    sigmaAngle2 = para.sigmaAngle2


    numCorres = candCorrs.shape[1]
    corIds1 = np.kron(np.arange(numCorres), np.ones((1,numCorres))).astype('int')
    corIds2 = np.kron(np.ones((1,numCorres)), np.arange(numCorres)).astype('int')
    #import pdb; pdb.set_trace()
    tp = corIds1 < corIds2
    corIds1 = corIds1[tp]
    corIds2 = corIds2[tp]


    if sourcePC.shape[1] != sourceNormal.shape[1]:
        corIds1_dis = corIds1[np.logical_and(corIds1 < corrs_num[1], corIds2 < corrs_num[1])]
        corIds2_dis = corIds2[np.logical_and(corIds1 < corrs_num[1], corIds2 < corrs_num[1])]
    else:
        corIds1_dis = corIds1
        corIds2_dis = corIds2
    
    temp = sourcePC[:, candCorrs[0,corIds1_dis].astype('int')] - sourcePC[:, candCorrs[0,corIds2_dis].astype('int')]
    sourceDis = np.sqrt(np.sum(np.power(temp, 2), axis=0))
    temp = targetPC[:, candCorrs[1,corIds1_dis].astype('int')] - targetPC[:, candCorrs[1,corIds2_dis].astype('int')]
    targetDis = np.sqrt(np.sum(np.power(temp, 2), axis=0))

    remainingIds = np.minimum(sourceDis, targetDis) > 0

    if sourcePC.shape[1] != sourceNormal.shape[1]:
        corIds1_dis = corIds1_dis[remainingIds]
        corIds2_dis = corIds2_dis[remainingIds]
        corIds1_rem = corIds1[np.logical_or(corIds1 >= corrs_num[1], corIds2 >= corrs_num[1])]
        corIds2_rem = corIds2[np.logical_or(corIds1 >= corrs_num[1], corIds2 >= corrs_num[1])]
        corIds1 = np.concatenate([corIds1_dis, corIds1_rem])
        corIds2 = np.concatenate([corIds2_dis, corIds2_rem])
    else:
        corIds1 = corIds1_dis[remainingIds]
        corIds2 = corIds2_dis[remainingIds]


    weightDis = np.abs(sourceDis[remainingIds] - targetDis[remainingIds])
    weightDis = np.exp(-np.power(weightDis,2) / (2*sigmaDist*sigmaDist))
    
    ids_1s = candCorrs[0, corIds1]
    ids_1t = candCorrs[1, corIds1]
    ids_2s = candCorrs[0, corIds2]
    ids_2t = candCorrs[1, corIds2] 
    
    vec_s = sourcePC[:, ids_1s.astype('int')] - sourcePC[:, ids_2s.astype('int')]
    vec_s = np.divide(vec_s, np.kron(np.ones((3,1)), np.sqrt(np.sum(np.power(vec_s,2), axis=0))))

    vec_t = targetPC[:, ids_1t.astype('int')] - targetPC[:, ids_2t.astype('int')]
    vec_t = np.divide(vec_t, np.kron(np.ones((3,1)), np.sqrt(np.sum(np.power(vec_t,2), axis=0))))

    angle1_s = np.arccos(np.sum(np.multiply(sourceNormal[:,ids_1s.astype('int')], sourceNormal[:,ids_2s.astype('int')]), axis=0).clip(-1,1))

    angle1_t = np.arccos(np.sum(np.multiply(targetNormal[:,ids_1t.astype('int')], targetNormal[:,ids_2t.astype('int')]), axis=0).clip(-1,1))
    
    angle2_s = np.arccos(np.sum(np.multiply(sourceNormal[:,ids_1s.astype('int')], vec_s), axis=0).clip(-1,1))
    angle2_t = np.arccos(np.sum(np.multiply(targetNormal[:,ids_1t.astype('int')], vec_t), axis=0).clip(-1,1))
    angle3_s = np.arccos(np.sum(np.multiply(sourceNormal[:,ids_2s.astype('int')], vec_s), axis=0).clip(-1,1))
    angle3_t = np.arccos(np.sum(np.multiply(targetNormal[:,ids_2t.astype('int')], vec_t), axis=0).clip(-1,1))
    
    angle1_dif = np.abs(angle1_s - angle1_t)
    angle2_dif = np.abs(angle2_s - angle2_t)
    angle3_dif = np.abs(angle3_s - angle3_t)

    weight_angle1 = np.exp(-np.power(angle1_dif,2)/(2*sigmaAngle1**2))
    weight_angle2 = np.exp(-np.power(angle2_dif,2)/(2*sigmaAngle2**2))
    weight_angle3 = np.exp(-np.power(angle3_dif,2)/(2*sigmaAngle2**2))

    #import pdb; pdb.set_trace()
    weight_geo = np.multiply(np.multiply(weightDis, weight_angle1), np.multiply(weight_angle2, weight_angle3))
    weightCorres = np.multiply(np.multiply(candCorrs[2, corIds1], candCorrs[2, corIds2]), weight_geo)
    #import pdb; pdb.set_trace()
    remainingIds = weightCorres > 0.1
    corIds1 = corIds1[remainingIds]
    corIds2 = corIds2[remainingIds]
    weightCorres = weightCorres[remainingIds]
    #import pdb; pdb.set_trace()
    


    matC = csc_matrix((weightCorres, (corIds1, corIds2)), shape=(numCorres, numCorres))
    matC = matC + matC.T

    return matC.toarray()

def leading_eigens(matC, numMatches):
    eigenVals, eigenVecs = sparse.linalg.eigs(matC, k=numMatches, which='LR')
    matches = [] # save ith correspondence pair

    for i in range(numMatches):
        if np.sum(eigenVecs[:,i]) < 0:
            eigenVecs[:, i] = -eigenVecs[:, i]
        
        order = np.argsort(-eigenVecs[:,i])
        ids = eigenVecs[order, i] > 0
        order = order[ids]

        consCorrIds = greedy_rounding(matC, order)

        matches.append(consCorrIds)

    return matches

def rigid_refinement(sourcePC, targetPC, sourceNormal, targetNormal, condCorrs, w_nor, w_fea, max_dist, spectral_method, sourceNum_list, targetNum_list):

    R_opt, t_opt = rigid_trans_regression(sourcePC, targetPC, sourceNormal, targetNormal, condCorrs, w_nor, sourceNum_list, targetNum_list)
    
    if 'irls' in spectral_method:
        # Peform reweighted least square
        R_opt, t_opt = generalized_icp(sourcePC, targetPC, sourceNormal, targetNormal, R_opt, t_opt, w_nor, w_fea, max_dist, sourceNum_list, targetNum_list)

    return R_opt, t_opt

def rigid_trans_regression(sourcePC, targetPC, sourceNormal, targetNormal, pointCorres, w_nor, sourceNum_list, targetNum_list):
    if pointCorres.shape[0] == 2:
        pointCorres = np.concatenate([pointCorres, np.ones((1, pointCorres.shape[1]))]).astype('int')


    pointCorres_nor = pointCorres
    #pointCorres_p = pointCorres[:,pointCorres[0,:].astype('int') < sourceNum_list[0]]
    pointCorres_p = pointCorres

    sourcePcs = sourcePC[:, pointCorres_p[0,:].astype('int')]
    sourceNors = sourceNormal[:, pointCorres[0,:].astype('int')]
    targetPcs = targetPC[:, pointCorres_p[1,:].astype('int')]
    targetNors = targetNormal[:, pointCorres[1,:].astype('int')]
    
    #import pdb; pdb.set_trace()

    weights_p = pointCorres_p[2,:]
    weights_nor = pointCorres_nor[2,:]

    sumOfWeights = np.sum(weights_p)
    #if sumOfWeights == 0:
    #    import pdb; pdb.set_trace()

    sourceCenter = np.sum((sourcePcs*np.kron(np.ones((3,1)),weights_p)).T, axis=0).T / sumOfWeights
    targetCenter = np.sum((targetPcs*np.kron(np.ones((3,1)),weights_p)).T, axis=0).T / sumOfWeights

    numS = sourcePcs.shape[1]
    numT = targetPcs.shape[1]

    sourcePcs = sourcePcs - np.matmul(sourceCenter[:,np.newaxis], np.ones((1,numS)))
    targetPcs = targetPcs - np.matmul(targetCenter[:,np.newaxis], np.ones((1,numT)))

    TP_p = np.kron(np.ones((3,1)), np.sqrt(weights_p))
    
    TP_nor = np.kron(np.ones((3,1)), np.sqrt(weights_nor))

    sourcePcs = np.concatenate([sourcePcs * TP_p, (np.sqrt(w_nor) * sourceNors) * TP_nor], axis=1)
    targetPcs = np.concatenate([targetPcs * TP_p, (np.sqrt(w_nor) * targetNors) * TP_nor], axis=1)
    if sumOfWeights == 0:
        sourcePcs = sourceNors
        targetPcs = targetNors

    S = np.matmul(sourcePcs, targetPcs.T)

    # notice that V = V.T in matlab!!!
    U, Sigma, V = np.linalg.svd(S)


    if np.linalg.det(S) > 0:
        R_opt = np.matmul(V.T, U.T)
    else:
        R_opt = np.matmul(np.matmul(V.T, np.diag([1,1,-1])), U.T)
   
    t_opt = targetCenter - np.matmul(R_opt, sourceCenter)

    return R_opt, t_opt


def generalized_icp(sourcePC, targetPC, sourceNormal, targetNormal, R_init, t_init, w_nor, w_fea, max_dist, sourceNum_list, targetNum_list):
    R_opt = R_init
    t_opt = t_init
    numS = sourcePC.shape[1]
    numT = targetPC.shape[1]

    sourceCombined = np.zeros((6, numS))
    sourceCombined[0:3,:] = sourcePC
    sourceCombined[3:6,:] = np.sqrt(w_nor)*sourceNormal

    targetCombined = np.zeros((6, numT))
    targetCombined[0:3,:] = targetPC
    targetCombined[3:6,:] = np.sqrt(w_nor)*targetNormal

    sourceCombined[0:3,:] = np.matmul(R_opt, sourcePC) + np.matmul(t_opt[:,np.newaxis], np.ones((1, numS)))
    sourceCombined[3:6,:] = np.sqrt(w_nor)*np.matmul(R_opt, sourceNormal)
    
    # first remove correspondence that are too far
    t_knn = NearestNeighbors(n_neighbors=1).fit(targetCombined.T)
    dist_st, IDX_st = t_knn.kneighbors(sourceCombined.T)

    s_knn = NearestNeighbors(n_neighbors=1).fit(sourceCombined.T)
    dist_ts, IDX_ts = s_knn.kneighbors(targetCombined.T)

    querySIds = np.where(dist_st.T < max_dist)[1]
    queryTIds = np.where(dist_ts.T < max_dist)[1]
    
    # sometimes there is no satisfying correspondence
    if len(querySIds) == 0 or len(queryTIds) == 0:
        return R_opt, t_opt

    for i in range(5):
        sourceCombined[0:3,:] = np.matmul(R_opt, sourcePC) + np.matmul(t_opt[:,np.newaxis], np.ones((1, numS)))
        sourceCombined[3:6,:] = np.sqrt(w_nor)*np.matmul(R_opt, sourceNormal)
        
        # weighted the remaining correspondence
        corres = corres_computation(sourceCombined, targetCombined, querySIds, queryTIds)

        R_opt, t_opt = rigid_trans_regression(sourcePC, targetPC, sourceNormal, targetNormal, corres, w_nor, sourceNum_list, targetNum_list)

    return R_opt, t_opt

def corres_computation(sourceCombined, targetCombined, querySIds, queryTIds):
    t_knn = NearestNeighbors(n_neighbors=1).fit(targetCombined.T)
    dist_st, IDX_st = t_knn.kneighbors(sourceCombined[:,querySIds].T)
    
    s_knn = NearestNeighbors(n_neighbors=1).fit(sourceCombined.T)
    dist_ts, IDX_ts = s_knn.kneighbors(targetCombined[:,queryTIds].T)
    
    corres = np.concatenate((np.concatenate((querySIds[np.newaxis,:], IDX_ts.T), axis=1), np.concatenate((IDX_st.T, queryTIds[np.newaxis,:]), axis=1)), axis=0)

    diff = sourceCombined[:, corres[0,:]]-targetCombined[:, corres[1,:]]
    sqrDis = np.sum(diff*diff, axis=0)
    sqrSigma = np.median(sqrDis)

    corres = np.concatenate((corres,np.exp(-sqrDis/(2*sqrSigma))[np.newaxis,:]), axis=0)

    return corres








