import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import numpy as np 
from RPModule.rpmodule import RelativePoseEstimation_helper,getMatchingPrimitive
from RPModule.rputil import opts
from util import angular_distance_np 
import argparse
import itertools
import util
from torch.utils.data import DataLoader
import os
import torch
from model.mymodel import SCNet
from utils import torch_op
import copy
import cv2

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--split', type=str,default='val',help='add identifier for this experiment')
parser.add_argument('--dataset', type=str,default='suncg',help='add identifier for this experiment')
parser.add_argument('--exp', type=str,default='sp_param',help='add identifier for this experiment')
parser.add_argument('--rm', action='store_true',help='add identifier for this experiment')
parser.add_argument('--maskMethod',type=str,default='second', help = 'suncg/matterport:second, scannet:kinect')
parser.add_argument('--alterStep', type=int,default=3,help='add identifier for this experiment')
parser.add_argument('--max_iter', type = int, default=30,help = 'max train step')

# net specification
parser.add_argument('--batchnorm', type = int, default = 1, help = 'whether to use batch norm in completion network') # 1
parser.add_argument('--useTanh', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--skipLayer', type = int, default = 1, help = 'whether to use skil connection in completion network') # 1
parser.add_argument('--outputType',type=str,default='rgbdnsf', help = 'types of output')
parser.add_argument('--snumclass',type=int,default=15, help = 'number of semantic class')
parser.add_argument('--featureDim',type=int,default=32, help = 'number of semantic class')

# pairwise match specification
parser.add_argument('--representation',type=str,default='skybox')
parser.add_argument('--completion', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--verbose', type = int, default = 0, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--rlevel', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
parser.add_argument('--para_init', type = str, default=None,help = 'whether to use tanh layer on feature maps')


parser.add_argument('--method',type=str,choices=['ours','ours_nc','ours_nr','super4pcs','fgs','gs','cgs'],default='ours',help='ours,super4pcs,fgs(fast global registration)')


# Siming add
parser.add_argument('--global_exp', type=str, default='./experiments/exp_', help='')    
parser.add_argument('--global_pretrain', type=str, default='./data/pretrained_model/360_image/', help='')
parser.add_argument('--gpu', type=str, default=None, help='')
parser.add_argument('--fitmethod', type=str, default='irls+sm', help='')
parser.add_argument('--old_scannet', type=int, default=0,help = 'old version of scannet')
parser.add_argument('--scannet_new_name', type=int, default=0,help = 'a tmp parameter to change the pose data path')
parser.add_argument('--idx', type=int, default=None)
args = parser.parse_args()


args.alterStep = 1 if args.method == 'ours_nr' else 3
args.completion = 0 if args.method == 'ours_nc' else 1
args.snumclass = 15 if 'suncg' in args.dataset else 21

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

scenes = glob.glob('./data/dataList/scannet_test_scenes/*')
scenes = ['./data/dataList/scannet_test_scenes/scene0084_00.npy', './data/dataList/scannet_test_scenes/scene0030_02.npy', './data/dataList/scannet_test_scenes/scene0193_01.npy', './data/dataList/scannet_test_scenes/scene0356_01.npy', './data/dataList/scannet_test_scenes/scene0474_01.npy', './data/dataList/scannet_test_scenes/scene0607_01.npy']
#import pdb; pdb.set_trace()
if args.idx is not None:
    chunk = 200
    aa = int(np.floor(len(scenes)/float(chunk))+1)
    scenes = [scenes[x] for x in range(aa*args.idx, min(len(scenes),aa*(args.idx+1)))]
    args.dataset = scenes[0].split('dataList/')[-1].split('.')[0]

# load network
net=SCNet(args).cuda()
if 'suncg' in args.dataset:
    checkpoint = torch.load(args.global_pretrain + '/suncg.comp.pth.tar')
elif 'matterport' in args.dataset:
    checkpoint = torch.load(args.global_pretrain + '/matterport.comp.pth.tar')
elif 'scannet' in args.dataset:
    checkpoint = torch.load(args.global_pretrain + '/scannet.comp.pth.tar')


state_dict = checkpoint['state_dict']
net.load_state_dict(state_dict)
net.cuda()
print("Successfully load SCNet weights!")

# load parameters for pairwise matching algo
if args.para_init is not None:
    para_val = np.loadtxt(args.para_init).reshape(-1,4)
else:
    para_val = np.array([0.523/2,0.523/2,0.08/2,0.01]).reshape(1,4)

args.para=opts(para_val[:,0],para_val[:,1],para_val[:,2],para_val[:,3])
args.para.method = args.fitmethod

if not os.path.exists("./data/relativePoseModule/"):
    os.makedirs("./data/relativePoseModule/")

# load dataset
if 'suncg' in args.dataset:
    #from datasets.SUNCG import SUNCG as Dataset
    from datasets.SUNCG_lmdb import SUNCG_lmdb as Dataset
    dataset_name='suncg'
    val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
        list_=f"./data/dataList/{args.dataset}.npy",singleView=0)
elif 'matterport' in args.dataset:
    from datasets.Matterport3D import Matterport3D as Dataset
    dataset_name='matterport'
    val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=True,normal=True,\
        list_=f"./data/dataList/{args.dataset}.npy",singleView=0)
elif 'scannet' in args.dataset:
    from datasets.ScanNet import ScanNet as Dataset
    #from datasets.ScanNet_lmdb import ScanNet_lmdb as Dataset
    dataset_name='scannet'
    val_dataset = Dataset(args.split, nViews=2,meta=False,rotate=False,rgbd=True,hmap=False,segm=False,normal=True,\
        list_=f"./data/dataList/{args.dataset}.npy",singleView=0,fullsize_rgbdn=True,\
        representation=args.representation, num_points=8192, scannet_new_name=args.scannet_new_name)


loader = DataLoader(val_dataset, batch_size=1, shuffle=False,num_workers=0,drop_last=True,collate_fn=util.collate_fn_cat, worker_init_fn=util.worker_init_fn)

# start test

loss = 0
angular_dis = 0

angular_dis_l = []
loss_l = []

import scipy.io as sio
pc_s = []
pc_t = []
normal_s = []
normal_t = []
feat_s = []
feat_t = []
weight_s = []
weight_t = []
r_s = []
r_t = []
path_s = []
path_t = []

overlaps = []

complete_depth_s = []
complete_depth_t = []
complete_normal_s = []
complete_normal_t = []


count = 0
count_= 0
def depth2pc(depth,needmask=False):
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

#import pdb; pdb.set_trace()
for i, data in enumerate(loader):
    print(count_) 
    # initialize data

    #lmdb preprocess
    if args.old_scannet==0:
        data['depth'] = data['depth'][:,:,0,:,:]
        rgb,depth,R,Q,norm,imgPath, overlap_val =data['rgb'],data['depth'],data['R'],data['Q'],data['norm'],data['imgsPath'], data['overlap']
        #data['depth'] = data['depth'][:,:,0,:,:]
    else:
        rgb,depth,R,Q,norm,imgPath = data['rgb'],data['depth'],data['R'],data['Q'],data['norm'],data['imgsPath']
    #import pdb; pdb.set_trace() 

    # use origin size scan for baselines on scannet dataset 
    if 'scannet' in args.dataset and 'ours' not in args.method:
        rgb,depth = data['rgb_full'], data['depth_full']
    #import pdb; pdb.set_trace()       
    R     = torch_op.npy(R)
    norm  = torch_op.npy(norm)
    rgb   = torch_op.npy(rgb*255).clip(0,255).astype('uint8')
    depth = torch_op.npy(depth)

    

    R_src = R[0,0,:,:]
    R_tgt = R[0,1,:,:]
    
    if 0:
    #if args.old_scannet == 0:
        #Rs = np.array([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
        Rs = np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
 
        # convert back to first view coordinate because SIFT keypoints are under the first view
        R_src_first = np.matmul(Rs, R_src)
        R_tgt_first = np.matmul(Rs, R_tgt)

        #R_gt_44 = np.matmul(R_tgt,np.linalg.inv(R_src))
        R_gt_44 = np.matmul(R_tgt_first, np.linalg.inv(R_src_first))
        R_gt = R_gt_44[:3,:3]
    else:
        R_src_first = R_src
        R_tgt_first = R_tgt

        #R_gt_44 = np.matmul(R_tgt,np.linalg.inv(R_src))
        R_gt_44 = np.matmul(R_tgt_first, np.linalg.inv(R_src_first))
        #R_gt_44 = np.matmul(R_tgt, np.linalg.inv(R_src))
        R_gt = R_gt_44[:3,:3]
    
    # generate source/target scans, point cloud

    depth_src,depth_tgt,_,_,color_src,color_tgt,pc_src,pc_tgt = util.parse_data(depth,rgb,norm,args.dataset,args.method)

    if len(pc_src) == 0 or len(pc_tgt)==0:
        print(f"this point cloud file contain no point")
        continue
            
    if args.old_scannet:
        
        overlap_val,cam_dist_this,pc_dist_this,pc_nn = util.point_cloud_overlap(pc_src, pc_tgt, R_gt_44)


        
    overlap = '0-0.1' if overlap_val <= 0.1 else '0.1-0.5' if overlap_val <= 0.5 else '0.5-1.0'


    data_s = {'rgb':   rgb[0,0,:,:,:].transpose(1,2,0),
            'depth': depth[0,0,:,:],
            'normal':norm[0,0,:,:,:].transpose(1,2,0),
            'R':     R_src_first}
    data_t = {'rgb':   rgb[0,1,:,:,:].transpose(1,2,0),
            'depth': depth[0,1,:,:],
            'normal':norm[0,1,:,:,:].transpose(1,2,0),
            'R':     R_tgt_first}




    args.idx_f_start = 0        
    if 'rgb' in args.outputType:
        args.idx_f_start += 3
    if 'n' in args.outputType:
        args.idx_f_start += 3
    if 'd' in args.outputType:
        args.idx_f_start += 1
    if 's' in args.outputType:
        args.idx_f_start += args.snumclass
    if 'f' in args.outputType:
        args.idx_f_end   = args.idx_f_start + args.featureDim
        

    with torch.set_grad_enabled(False):
        R_hat=np.eye(4)
            
        # get the complete scans


        complete_s=torch.cat((torch_op.v(data['rgb'][:,0,:,:,:]),torch_op.v(data['norm'][:,0,:,:,:]),torch_op.v(data['depth'][:,0:1,:,:])),1)
        complete_t=torch.cat((torch_op.v(data['rgb'][:,1,:,:,:]),torch_op.v(data['norm'][:,1,:,:,:]),torch_op.v(data['depth'][:,1:2,:,:])),1)

        

        # apply the observation mask
        view_s,mask_s,_ = util.apply_mask(complete_s.clone(),args.maskMethod)
        view_t,mask_t,_ = util.apply_mask(complete_t.clone(),args.maskMethod)
        mask_s=torch_op.npy(mask_s[0,:,:,:]).transpose(1,2,0)
        mask_t=torch_op.npy(mask_t[0,:,:,:]).transpose(1,2,0)

        # append mask for valid data
        tpmask = (view_s[:,6:7,:,:]!=0).float().cuda()
        view_s=torch.cat((view_s,tpmask),1)
        tpmask = (view_t[:,6:7,:,:]!=0).float().cuda()
        view_t=torch.cat((view_t,tpmask),1)


        # warp the second scan using current transformation estimation
        view_t2s=torch_op.v(util.warping(torch_op.npy(view_t),np.linalg.inv(R_hat),args.dataset))
        view_s2t=torch_op.v(util.warping(torch_op.npy(view_s),R_hat,args.dataset))
        # append the warped scans
        #view0 = torch.cat((view_s,view_t2s),1)
        #view1 = torch.cat((view_t,view_s2t),1)

        view0 = torch.cat((view_s,view_s),1)
        view1 = torch.cat((view_t,view_t),1)
        # generate complete scans
        f=net(torch.cat((view0,view1)))
        f0=f[0:1,:,:,:]
        f1=f[1:2,:,:,:]
        
        data_sc,data_tc={},{}

        complete_normal_s.append(torch_op.npy(f0[0,3:6,:,:]))
        complete_normal_t.append(torch_op.npy(f1[0,3:6,:,:]))

        complete_depth_s.append(torch_op.npy(f0[0,6,:,:]))
        complete_depth_t.append(torch_op.npy(f1[0,6,:,:]))


        # replace the observed region with gt depth/normal
        data_sc['normal'] = (1-mask_s)*torch_op.npy(f0[0,3:6,:,:]).transpose(1,2,0)+mask_s*data_s['normal']
        data_tc['normal'] = (1-mask_t)*torch_op.npy(f1[0,3:6,:,:]).transpose(1,2,0)+mask_t*data_t['normal']
        data_sc['normal']/= (np.linalg.norm(data_sc['normal'],axis=2,keepdims=True)+1e-6)
        data_tc['normal']/= (np.linalg.norm(data_tc['normal'],axis=2,keepdims=True)+1e-6)
        
        data_sc['depth']  = (1-mask_s[:,:,0])*torch_op.npy(f0[0,6,:,:])+mask_s[:,:,0]*data_s['depth']
        data_tc['depth']  = (1-mask_t[:,:,0])*torch_op.npy(f1[0,6,:,:])+mask_t[:,:,0]*data_t['depth']        
        data_sc['obs_mask']   = mask_s.copy()
        data_tc['obs_mask']   = mask_t.copy()


        data_sc['rgb']    = (mask_s*data_s['rgb']).astype('uint8')
        data_tc['rgb']    = (mask_t*data_t['rgb']).astype('uint8')
        # for scannet, we use the original size rgb image(480x640) to extract sift keypoint
        if 'scannet' in args.dataset:
            data_sc['rgb_full'] = (torch_op.npy(data['rgb_full'][0,0,:,:,:])*255).astype('uint8') 
            data_tc['rgb_full'] = (torch_op.npy(data['rgb_full'][0,1,:,:,:])*255).astype('uint8')
            data_sc['depth_full'] = torch_op.npy(data['depth_full'][0,0,:,:])
            data_tc['depth_full'] = torch_op.npy(data['depth_full'][0,1,:,:])

                
        # extract feature maps
        f0_feat=f0[:,args.idx_f_start:args.idx_f_end,:,:]
        f1_feat=f1[:,args.idx_f_start:args.idx_f_end,:,:]
        data_sc['feat']=f0_feat.squeeze(0)
        data_tc['feat']=f1_feat.squeeze(0)


        # extract matching primitive from image representation
        #import pdb; pdb.set_trace()        
        pts3d,ptt3d,ptsns,ptsnt,dess,dest,ptsW,pttW = getMatchingPrimitive(data_sc,data_tc,dataset_name,args.representation,args.completion)
        
        # early return if too few keypoint detected
        if pts3d is None or ptt3d is None or pts3d.shape[0]<2 or pts3d.shape[0]<2:
            continue       
        
        if 1:

            # source filter
            path_s.append(imgPath[0])
            pts3d_t = pts3d.T
            pts3d_max = np.max(pts3d_t, axis=1)
            nonzero_idx = (pts3d_max != 0)
            pts3d_t = pts3d_t[nonzero_idx, :]
            ptsns = ptsns[nonzero_idx, :]
            dess = dess[nonzero_idx, :]
            

            # target filter
            path_t.append(imgPath[1])
            ptt3d_t = ptt3d.T
            ptt3d_max = np.max(ptt3d_t, axis=1)
            nonzero_idx = (ptt3d_max != 0)
            ptt3d_t = ptt3d_t[nonzero_idx, :]
            ptsnt = ptsnt[nonzero_idx, :]
            dest = dest[nonzero_idx, :]

            pc_s.append(pts3d_t)
            pc_t.append(ptt3d_t)


            normal_s.append(ptsns)
            normal_t.append(ptsnt)
            feat_s.append(dess)
            feat_t.append(dest)

            r_s.append(R_src_first)
            r_t.append(R_tgt_first)


            overlaps.append(overlap_val)

        
        if i > 0 and i // 100 == 0:

            sio.savemat('./data/test_data/suncg_source_100.mat',{'pc': np.asarray(pc_s), 'normal': np.asarray(normal_s), 'feat': np.asarray(feat_s), 'R': np.asarray(r_s), 'overlap': np.asarray(overlaps), 'path': path_s})
            sio.savemat('./data/test_data/suncg_target_100.mat',{'pc': np.asarray(pc_t), 'normal': np.asarray(normal_t), 'feat': np.asarray(feat_t), 'R': np.asarray(r_t), 'overlap': np.asarray(overlaps), 'path': path_t})

            import pdb; pdb.set_trace()

        
        count_ += 1
        # get plane features

        #pts3d_pl, ptt3d_pl, ptsns_pl, pttnt_pl, ptsW_pl, pttW_pl, dess_pl, dest_pl = util.render_plane_point(R_src[:3,:3], R_tgt[:3,:3])
        #import pdb; pdb.set_trace() 


        #import pdb; pdb.set_trace()
        '''


        para_this = copy.copy(args.para)
        para_this.sigmaAngle1 = para_this.sigmaAngle1
        para_this.sigmaAngle2 = para_this.sigmaAngle2
        para_this.sigmaDist = para_this.sigmaDist
        para_this.sigmaFeat = para_this.sigmaFeat

        #pts3d = np.concatenate([pts3d, pts3d_pl], 1)
        #ptt3d = np.concatenate([ptt3d, ptt3d_pl], 1)
        #ptsns = np.concatenate([ptsns, ptsns_pl.T], 0)
        #ptsnt = np.concatenate([ptsnt, pttnt_pl.T], 0)
        #dess = np.concatenate([dess, dess_pl], 0)
        #dest = np.concatenate([dest, dest_pl], 0)
        #ptsW = np.concatenate([ptsW, ptsW_pl], 0)
        #pttW = np.concatenate([pttW, pttW_pl], 0)


        R_hat = RelativePoseEstimation_helper({'pc':pts3d.T,'normal':ptsns,'feat':dess,'weight':ptsW},{'pc':ptt3d.T,'normal':ptsnt,'feat':dest,'weight':pttW},para_this)


        # calculate loss

        loss += np.power(R_hat[:3,:3] - R_gt[:3,:3],2).sum()
        angular_dis_cur = angular_distance_np(R_hat[:3,:3].reshape(1,3,3),R_gt[:3,:3].reshape(1,3,3))[0]
        angular_dis += angular_dis_cur
        angular_dis_l.append(angular_dis_cur)
        count += 1
        print(i)
        print(angular_dis/count)


        '''



        













