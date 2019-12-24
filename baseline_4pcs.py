from open3d import *
import util
import time
import numpy as np
from subprocess import STDOUT, check_output
import copy
import uuid
import logging
logger = logging.getLogger(__name__)
import argparse
import scipy.io as sio
import os
import cv2
import baselines
import scipy.io as sio


def LoadImage(PATH,depth=True):
    #import pdb; pdb.set_trace() 
    if depth:
        img = cv2.imread(PATH,2) / 1000.
    else:
        img = cv2.imread(PATH)

    return img

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




def _parse_args():
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--dataset', type = str, default = 'scannet', help = 'options: suncg, scannet')
    parser.add_argument('--save_rp', type = int, default = 0, help = '')
    parser.add_argument('--method', type = str, default = '4pcs', help = 'option 4pcs, gr')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = _parse_args()

    if args.dataset == 'scannet':
        dataS = sio.loadmat('./data/test_data/scannet_source_clean_v5_top5.mat')
        dataT = sio.loadmat('./data/test_data/scannet_target_clean_v5_top5.mat')
    elif args.dataset == 'suncg':
        dataS = sio.loadmat('./data/test_data/suncg_source_clean_v4_top5.mat')
        dataT = sio.loadmat('./data/test_data/suncg_target_clean_v4_top5.mat')

    num_data = dataS['R'].shape[0]

    non_error = 0
    large_error = 0
    non_data = 0
    large_data = 0
    non_trans_error = 0
    large_trans_error = 0
    

    pred_pose_l = []
    gt_pose_l = []
    path_s_l = []
    path_t_l = []

    for i in range(num_data):
        print(i)
        path_s = dataS['path'][i][0]
        path_t = dataT['path'][i][0]
        path_s_l.append(path_s)
        path_t_l.append(path_t)

        base_s = '/'.join(path_s.split('/')[:-1])
        base_t = '/'.join(path_t.split('/')[:-1])

        scan_s = path_s.split('/')[-1]
        scan_t = path_t.split('/')[-1]
        
        if args.dataset == 'suncg':
            #import pdb; pdb.set_trace()
            base_path = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/SkyBox/SUNCG/'
            base_s = base_path + base_s.split('SUNCG/')[-1]
            base_t = base_path + base_t.split('SUNCG/')[-1]

        depth_s = LoadImage(os.path.join(base_s, 'depth', scan_s.strip(' ') + '.png'))
        depth_t = LoadImage(os.path.join(base_t, 'depth', scan_t.strip(' ') + '.png'))

        #rgb_s = LoadImage(os.path.join(base_s, 'rgb', scan_s + '.png'),depth=False)/255.
        #rgb_t = LoadImage(os.path.join(base_t, 'rgb', scan_t + '.png'),depth=False)/255.

        #norm_s = LoadImage(os.path.join(base_s, 'normal', scan_s + '.png'),depth=False)/255.*2-1
        #norm_t = LoadImage(os.path.join(base_t, 'normal', scan_t + '.png'),depth=False)/255.*2-1

        pc_s,mask_s = depth2pc(depth_s[80-33:80+33,160+80-44:160+80+44], 'scannet')
        pc_t,mask_t = depth2pc(depth_t[80-33:80+33,160+80-44:160+80+44], 'scannet')

        #pc_s_c = rgb_s[80-33:80+33,160+80-44:160+80+44].reshape(-1,3)[mask_s]
        #pc_t_c = rgb_t[80-33:80+33,160+80-44:160+80+44].reshape(-1,3)[mask_t]


        #pc_s_n = norm_s[80-33:80+33,160+80-44:160+80+44].reshape(-1,3)[mask_s]
        #pc_t_n = norm_t[80-33:80+33,160+80-44:160+80+44].reshape(-1,3)[mask_t]


        gt_pose = dataS['gt_pose'][i]

        pred_pose = dataS['pred_pose'][i]

        pred_num = len(pred_pose)
            


        #save_path = '-'.join([base_s.split('/')[-1], scan_s, scan_t]) 

        #overlap_val,_,_,_ = util.point_cloud_overlap(pc_s, pc_t, gt_pose)
        overlap_val = dataS['overlap'][0][i]

        
        if args.method == '4pcs':
            pred_pose = baselines.super4pcs(pc_s, pc_t)
        elif args.method == 'gr':
            pred_pose = baselines.open3d_global_registration(pc_s, pc_t)

        pred_pose_l.append(pred_pose)
        gt_pose_l.append(gt_pose)

        ad_tmp = util.angular_distance_np(pred_pose[:3,:3].reshape(1,3,3), gt_pose[:3,:3].reshape(1,3,3))[0]
        trans_error = np.linalg.norm(pred_pose[:3,3] - gt_pose[:3,3])
        if overlap_val <= 0.1:
            non_error += ad_tmp
            non_data += 1
            non_trans_error += trans_error 
        else:
            large_error += ad_tmp
            large_data += 1
            large_trans_error += trans_error

    print("0.1-1.0 data number:", large_data)
    print("0.1-1.0 overlap rotation error:", float(large_error) / large_data)
    print("0.1-1.0 overlap translation error:", float(large_trans_error) / large_data)
    print("0-0.1 data number:", non_data)
    print("0-0.1 overlap rotation error:", float(non_error) / non_data)
    print("0-0.1 overlap translation error:", float(non_trans_error) / non_data)

    if args.save_rp == 1:
        if args.method == '4pcs':
            appending = '4pcs'
        elif args.method == 'gr':
            appending = 'gr'

        sio.savemat('./data/test_data/suncg_source_clean_v4_%s.mat' % appending,{'path': path_s_l, 'pred_pose': pred_pose_l, 'gt_pose': gt_pose_l})
        sio.savemat('./data/test_data/suncg_target_clean_v4_%s.mat' % appending,{'path': path_t_l, 'pred_pose': pred_pose_l, 'gt_pose': gt_pose_l})















