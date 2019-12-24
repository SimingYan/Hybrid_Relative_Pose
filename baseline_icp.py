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
import statistics

def general_icp(pc_s, pc_t, trans_init):

    pcd_s = geometry.PointCloud()
    pcd_t = geometry.PointCloud()
    pcd_s.points = utility.Vector3dVector(pc_s)
    pcd_t.points = utility.Vector3dVector(pc_t)
    reg_p2p = registration.registration_icp(pcd_s, pcd_t, 0.02, trans_init)

    return reg_p2p.transformation


def LoadImage(PATH,depth=True):
    
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
    parser.add_argument('--dataset', type = str, default = 'scannet', help = 'options: suncg, scannet, matterport')
    parser.add_argument('--global_method', type = str, default = 'ours', help = 'options:  ours, gr')
    parser.add_argument('--top', type = int, default = 1, help = 'options: 1,3,5')
    parser.add_argument('--tolerate_error', type = float, default = 3.0, help = '')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = _parse_args()

    if args.dataset == 'scannet':

        if args.global_method == 'ours':
            dataS = sio.loadmat('./data/test_data/scannet_source.mat')
            dataT = sio.loadmat('./data/test_data/scannet_target.mat')
        elif args.global_method == 'gr':
            dataS = sio.loadmat('./data/test_data/scannet_source_gr.mat')
            dataT = sio.loadmat('./data/test_data/scannet_target_gr.mat')

    elif args.dataset == 'suncg':
        if args.global_method == 'ours':
            dataS = sio.loadmat('./data/test_data/suncg_source.mat')
            dataT = sio.loadmat('./data/test_data/suncg_target.mat')
        elif args.global_method == 'gr':
            dataS = sio.loadmat('./data/test_data/suncg_source_gr.mat')
            dataT = sio.loadmat('./data/test_data/suncg_target_gr.mat')

    elif args.dataset == 'matterport':
        if args.global_method == 'ours':
            dataS = sio.loadmat('./data/test_data/matterport_source.mat')
            dataT = sio.loadmat('./data/test_data/matterport_target.mat')
        elif args.global_method == 'gr':
            dataS = sio.loadmat('./data/test_data/matterport_source_gr.mat')
            dataT = sio.loadmat('./data/test_data/matterport_target_gr.mat')

    num_data = dataS['gt_pose'].shape[0]

    average_error = 0
    correct_num = 0
    large_overlap = 0
    error_list = []
    trans_list = []
    average_trans = 0
    for i in range(num_data):
        print(i)
        #import pdb; pdb.set_trace()
        if args.global_method == 'ours':
            path_s = dataS['path'][i][0]
            path_t = dataT['path'][i][0]
        elif args.global_method == 'gr':
            path_s = dataS['path'][i]
            path_t = dataT['path'][i]
        
        base_s = '/'.join(path_s.split('/')[:-1])
        base_t = '/'.join(path_t.split('/')[:-1])

        scan_s = path_s.split('/')[-1].strip(' ')
        scan_t = path_t.split('/')[-1].strip(' ')

        if args.dataset == 'suncg':
            #import pdb; pdb.set_trace()
            base_path = '/media/yzp12/wdblue/2020_CVPR_Hybrid/data/SkyBox/SUNCG/'
            base_s = base_path + base_s.split('SUNCG/')[-1]
            base_t = base_path + base_t.split('SUNCG/')[-1]
        #import pdb; pdb.set_trace()
        depth_s = LoadImage(os.path.join(base_s, 'depth', scan_s.strip(' ') + '.png'))
        depth_t = LoadImage(os.path.join(base_t, 'depth', scan_t.strip(' ') + '.png'))
        
        if args.dataset == 'suncg' or args.dataset == 'matterport':
            pc_s,mask_s = depth2pc(depth_s[:,160:160*2], args.dataset)
            pc_t,mask_t = depth2pc(depth_t[:,160:160*2], args.dataset)
        else:
            pc_s,mask_s = depth2pc(depth_s[80-33:80+33,160+80-44:160+80+44], args.dataset)
            pc_t,mask_t = depth2pc(depth_t[80-33:80+33,160+80-44:160+80+44], args.dataset)


        gt_pose = dataS['gt_pose'][i]
        
        if args.global_method == 'ours':
            pred_pose = dataS['pred_pose'][i]
        elif args.global_method == 'gr':
            pred_pose = dataS['pred_pose'][i][np.newaxis,:]
        pred_num = args.top
        
        if args.global_method == 'ours':
            overlap_val = dataS['overlap'][0][i]

            if overlap_val > 0.1:
                 large_overlap += 1

        min_ad = 1000
        for j in range(pred_num):
            current_pose = pred_pose[j]

            icp_pose = general_icp(pc_s, pc_t, current_pose)

            ad_tmp = util.angular_distance_np(icp_pose[:3,:3].reshape(1,3,3), gt_pose[:3,:3].reshape(1,3,3))[0]
            
            if ad_tmp < min_ad:
                min_ad = ad_tmp
             
            if ad_tmp < args.tolerate_error:
                correct_num += 1
                break

        trans_error = np.linalg.norm(icp_pose[:3,3] - gt_pose[:3,3])

        average_trans += trans_error
        average_error += min_ad

        trans_list.append(trans_error)
        error_list.append(min_ad)

    print("correct percentage after applying icp:", float(correct_num) / num_data)
    print("percentage of large overlap data:", float(large_overlap) / num_data)
    print("average rotation error:", average_error / num_data)
    print("median rotation error:", statistics.median(error_list))
    print("average translation error:", average_trans / num_data)
    print("median translation error:", statistics.median(trans_list))
















