from open3d import *

import copy
from progress.bar import Bar
import config
import os
import numpy as np

import cv2
import matplotlib.pyplot as plt

import util

from RPModule.rputil import opts
import argparse

import time

import logging
import scipy.io as sio
from spectral_module import Spectral_Matching, Spectral_Matching_M
from RPModule.rputil import opts
from Hybrid_RelativePose import Hybrid_Spectral_Matching, horn87_np, Hybrid_Spectral_Matching_M
import glob

def extract_id(n):
    data_id = n[0].split('/')[-1]
    return int(data_id)

def extend_data(dataS):
    new_dataS = {}
    new_dataT = {}
    new_dataS['pc'] = []
    new_dataS['normal'] = []
    new_dataS['feat'] = []
    new_dataS['R'] = []
    new_dataS['overlap'] = []
    new_dataS['path'] = []
    new_dataT['pc'] = []
    new_dataT['normal'] = []
    new_dataT['feat'] = []
    new_dataT['R'] = []
    new_dataT['overlap'] = []
    new_dataT['path'] = []
    
    new_dataS['dense_depth'] = []
    new_dataT['dense_depth'] = []
    new_dataS['dense_normal'] = []
    new_dataT['dense_normal'] = []

    data_ids = list(map(extract_id, list(dataS['path'])))
    index_ids = list(range(0, len(dataS['path'])))
 
    data_ids[:], index_ids[:] = zip(*sorted(zip(data_ids,index_ids)))

    interval = int(len(data_ids) / 30)

    # for test, extract data every w0 data
    index_ids = index_ids[::interval] 
 
    
    from itertools import permutations
    for p in permutations(index_ids, 2):
        new_dataS['pc'].append(dataS['pc'][0][p[0]])
        new_dataS['normal'].append(dataS['normal'][0][p[0]])
        new_dataS['feat'].append(dataS['feat'][0][p[0]])
        new_dataS['path'].append(dataS['path'][p[0]])
        new_dataS['R'].append(dataS['R'][p[0]])
        new_dataS['dense_depth'].append(dataS['dense_depth'][p[0]])
        new_dataS['dense_normal'].append(dataS['dense_normal'][p[0]])

        new_dataT['pc'].append(dataS['pc'][0][p[1]])
        new_dataT['normal'].append(dataS['normal'][0][p[1]])
        new_dataT['feat'].append(dataS['feat'][0][p[1]])
        new_dataT['path'].append(dataS['path'][p[1]])
        new_dataT['R'].append(dataS['R'][p[1]])
        new_dataT['dense_depth'].append(dataS['dense_depth'][p[1]])
        new_dataT['dense_normal'].append(dataS['dense_normal'][p[1]])

        R_gt_44 = np.matmul(dataS['R'][p[1]], np.linalg.inv(dataS['R'][p[0]]))
        overlap_val,_,_,_ = util.point_cloud_overlap(dataS['pc'][0][p[0]], dataS['pc'][0][p[1]], R_gt_44)
        #import pdb; pdb.set_trace()
        new_dataS['overlap'].append(overlap_val)
        new_dataT['overlap'].append(overlap_val)

    
    new_dataS['pc'] = np.asarray(new_dataS['pc'])
    new_dataS['normal'] = np.asarray(new_dataS['normal'])
    new_dataS['feat'] = np.asarray(new_dataS['feat'])
    new_dataS['R'] = np.asarray(new_dataS['R'])
    new_dataS['overlap'] = np.asarray(new_dataS['overlap'])
    new_dataS['path'] = np.asarray(new_dataS['path'])
    new_dataS['dense_depth'] = np.asarray(new_dataS['dense_depth'])
    new_dataS['dense_normal'] = np.asarray(new_dataS['dense_normal'])

    new_dataT['pc'] = np.asarray(new_dataT['pc'])
    new_dataT['normal'] = np.asarray(new_dataT['normal'])
    new_dataT['feat'] = np.asarray(new_dataT['feat'])
    new_dataT['R'] = np.asarray(new_dataT['R'])
    new_dataT['overlap'] = np.asarray(new_dataT['overlap'])
    new_dataT['path'] = np.asarray(new_dataT['path'])
    new_dataT['dense_depth'] = np.asarray(new_dataT['dense_depth'])
    new_dataT['dense_normal'] = np.asarray(new_dataT['dense_normal'])

    return new_dataS, new_dataT

def getTestdata(args, name=None):
    if 'suncg' in args.dataList:
        dataset_name = 'suncg'

        dataS = sio.loadmat('./data/eval/test_data/suncg_source.mat')
        dataT = sio.loadmat('./data/eval/test_data/suncg_target.mat')

    elif 'scannet' in args.dataList:
        dataset_name = 'scannet'
        if name is not None:
            dataS = sio.loadmat('./data/eval/test_data/scannet_test_scenes_full/'+name+'_source.mat')
            dataT = sio.loadmat('./data/eval/test_data/scannet_test_scenes_full/'+name+'_target.mat')
        else:
            dataS = sio.loadmat('./data/eval/test_data/scannet_source.mat')
            dataT = sio.loadmat('./data/eval/test_data/scannet_target.mat')

    elif 'matterport' in args.dataList:
        dataset_name = 'matterport'

        dataS = sio.loadmat('./data/eval/test_data/matterport_source.mat')
        dataT = sio.loadmat('./data/eval/test_data/matterport_target.mat')
    return dataset_name, dataS, dataT

def _parse_args():
    
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--dataList', type = str, default = 'matterport3dv1', help = 'options: suncgv3,scannetv1,matterport3dv1')

    parser.add_argument('--sigmaFeat',type=float, default=0.01, help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--maxIter',type=int,default=10000000, help = 'number of pairs to be tested')
    parser.add_argument('--outputType',type=str,default='rgbdnsf', help = 'types of output')
    parser.add_argument('--debug',action='store_true', help = 'for debug')
    parser.add_argument('--exp',type=str,default='', help = 'will create a folder with such name under experiments/')
    parser.add_argument('--snumclass',type=int,default=15, help = 'number of semantic class')
    parser.add_argument('--featureDim',type=int,default=32, help = 'feature dimension')
    parser.add_argument('--maskMethod',type=str,default='second',help='observe the second view')
    parser.add_argument('--d',type=str,default='', help = '')
    parser.add_argument('--entrySplit',type=int,default=None, help = 'use for parallel eval')
    parser.add_argument('--representation',type=str,default='skybox')
    parser.add_argument('--method',type=str,choices=['ours','ours_nc','ours_nr','super4pcs','fgs','gs','cgs'],default='ours',help='ours,super4pcs,fgs(fast global registration)')
    parser.add_argument('--useTanh', type = int, default = 1, help = 'whether to use tanh layer on feature maps')
    parser.add_argument('--saveCompletion', type = int, default = 1, help = 'save the completion result')
    parser.add_argument('--batchnorm', type = int, default = 1, help = 'whether to use batch norm in completion network')
    parser.add_argument('--skipLayer', type = int, default = 1, help = 'whether to use skil connection in completion network')
    parser.add_argument('--num_repeat', type = int, default = 1, help = 'repeat times')
    parser.add_argument('--rm',action='store_true',help='will remove previous evaluation named args.exp')
    parser.add_argument('--para', type = str, default=None,help = 'file specify parameters for pairwise matching module')
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], help="Set the logging level")

    # Siming add
    parser.add_argument('--global_exp', type=str, default='./experiments/exp_', help='')    
    parser.add_argument('--global_pretrain', type=str, default='./data/pretrained_model', help='')
    parser.add_argument('--fitmethod', type=str, default='original', help='')
    parser.add_argument('--numMatches', type=int, default=6, help='the number of output relative pose')
    parser.add_argument('--hybrid', type=int, default=0, help='whether use hybrid representation')
    parser.add_argument('--hybrid_method', type=str, default='360', help='which hybrid representation are you using')
    parser.add_argument('--w_plane_1', type=float, default=2.0, help='weight of plane correspondence')
    parser.add_argument('--w_plane_2', type=float, default=1.0, help='weight of plane correspondence pair, should be greater than 1')
    parser.add_argument('--filename', type=str, default=None, help='')
    parser.add_argument('--perturb_rate', type=float, default=None, help='perturb the plane point to search the predict range')
    parser.add_argument('--w_topdown', type=float, default=1.0, help='weight of topdown correspondence')
    parser.add_argument('--sigmaDist', type=float, default=0.3, help='weight of topdown correspondence')
    parser.add_argument('--sigmaAngle1',type=float, default=0.5236,help = 'parameter for our pairwise matching algorithm')
    parser.add_argument('--sigmaAngle2',type=float, default=0.5236, help = 'parameter for our pairwise matching algorithm')

    parser.add_argument('--sample_data', type=int, default=0, help='sample 100 test data')
    parser.add_argument('--save_data', type=int, default=0, help='sample 100 test data')
    parser.add_argument('--save_primitive', type=int, default=0, help='sample 100 test data')
    parser.add_argument('--save_rp', type=int, default=0, help='sample 100 test data')
    parser.add_argument('--print_each', type=int, default=0, help='sample 100 test data')
    parser.add_argument('--draw_corres', type=int, default=0, help='draw correspondence')

    parser.add_argument('--detect_each', type=int, default=0, help='detect each 20 data')
    parser.add_argument('--idx', type=int, default=None, help='used for parallel')   
    parser.add_argument('--parallel', type=int, default=0, help='used for parallel')

    args = parser.parse_args()
    if args.d: os.environ["CUDA_VISIBLE_DEVICES"] = args.d
    args.alterStep = 1 if args.method == 'ours_nr' else 3
    args.completion = 0 if args.method == 'ours_nc' else 1
    args.snumclass = 15 if 'suncg' in args.dataList else 21
    if args.logLevel:
        logging.basicConfig(level=getattr(logging, args.logLevel))
    


    print("\n parameters... *******************************\n")
    print(f"evaluate on {args.dataList}")
    print(f"using method: {args.method}")
    print(f"mask method: {args.maskMethod}")
    if 'ours' in args.method:
        print(f"output type: {args.outputType}")
        print(f"semantic classes: {args.snumclass}")
        print(f"feature dimension: {args.featureDim}")
        print(f"skipLayer: {args.skipLayer}")
        print(f"fit method: {args.fitmethod}")
    print("\n parameters... *******************************\n")
    time.sleep(5)


    args.rpm_para = opts()
    
    args.perStepPara = False
    if args.para is not None:
        para_val = np.loadtxt(args.para).reshape(-1,4)
        args.rpm_para.sigmaAngle1 = para_val[:,0]
        args.rpm_para.sigmaAngle2 = para_val[:,1]
        args.rpm_para.sigmaDist = para_val[:,2]
        args.rpm_para.sigmaFeat = para_val[:,3]
        args.perStepPara = True
    else:
        if args.sigmaAngle1: args.rpm_para.sigmaAngle1 = args.sigmaAngle1
        if args.sigmaAngle2: args.rpm_para.sigmaAngle2 = args.sigmaAngle2
        if args.sigmaDist: args.rpm_para.sigmaDist = args.sigmaDist
        if args.sigmaFeat: args.rpm_para.sigmaFeat = args.sigmaFeat

    return args

if __name__ == '__main__':
    
    args = _parse_args()
    para = opts()
    log = logging.getLogger(__name__)
    
    # test ad 
    ad_test = 0


    if not os.path.exists("tmp/rpe"):
        os.makedirs("tmp/rpe")
    exp_dir = f"tmp/rpe/{args.exp}"
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    primitive_file = f"./data/hybrid/{args.dataList}_middleoverlap.npy"
    primitives = []

    if os.path.exists('/scratch'):
        scenes = glob.glob('./data/test_data/scannet_test_scenes_full/*_source.mat')

    if args.idx is not None:
        chunk = 250
        aa = int(np.floor(len(scenes)/float(chunk))+1)
        scenes = [scenes[x] for x in range(aa*args.idx, min(len(scenes),aa*(args.idx+1)))]
        scene_id = scenes[0].split('test_scenes_full/')[-1].split('_source')[0]
        print("scene_id:", scene_id)
        #scene_id = ['scene0011_00'
        dataset_name, dataS, dataT = getTestdata(args, name=scene_id)


        if os.path.exists('/scratch/cluster/yzp12/projects/2020_CVPR_Hybrid/third_party/Hybrid_Representation/RelativePose/data/test_data/scannet_each_scenes_rp_hybrid_3_v2/' + scene_id + '_source.mat'):
            print("Has existed!")
            exit()
    else:
        # get 360-image matlab data
        dataset_name, dataS, dataT = getTestdata(args)

    if args.parallel == 1:
        dataS, dataT = extend_data(dataS)

    if args.parallel:
        args.filename = './data/test_data/scannet_each_scenes_rp_hybrid_3_v2/%s.txt' % scene_id
    if args.filename is not None:
        f = open(args.filename, 'a')
        print("w_plane_1:{}, w_plane_2:{} w_topdown:{}\n".format(args.w_plane_1,args.w_plane_2, args.w_topdown), file=f)

    num_data = dataS['R'].shape[0] # data number

    bar = Bar('Progress', max=num_data)

    speedBenchmark=[]

    #if 'matterport' in args.dataList:
    if 0:
        Overlaps = ['0-0.1','0.1-0.5','0.5-1.0']
    else:
        Overlaps = ['0-0.1','0.1-1.0']

    adstatsOverlaps = {it:[] for it in Overlaps} # angular distance result based on overlap
    transstatsOverlaps = {it:[] for it in Overlaps} # translation error
    error_stats=[]
    


    n_run = len(error_stats)//100
    args.num_repeat -= n_run

    para.numMatches = args.numMatches    
    para.sigmaDist = args.sigmaDist
    para.sigmaAngle1 = args.sigmaAngle1
    para.sigmaAngle2 = args.sigmaAngle2
    para.draw_corres = args.draw_corres

    non_overlap_len = 0
    small_overlap_len = 0
    large_overlap_len = 0

        
    if args.filename is not None:
        print("num_data:{}\n".format(num_data), file=f)

    print(num_data)
    
    no_plane = 0
    no_topdown = 0
    best_distribution = [0,0,0,0,0] 
    # just for sample good data for paper showing

    good_id = []

    for j in range(num_data):
        st = time.time()
        np.random.seed()

        
        R_gt_44 = np.matmul(dataT['R'][j], np.linalg.inv(dataS['R'][j]))
        R_gt = R_gt_44[:3,:3]
        
        # source domain data

        if args.parallel == 0:
            dataS_tmp = {}
            dataS_tmp['pc'] = dataS['pc'][0][j]
            dataS_tmp['normal'] = dataS['normal'][0][j]
            dataS_tmp['feat'] = dataS['feat'][0][j]
             
            # target domain data
            dataT_tmp = {}
            dataT_tmp['pc'] = dataT['pc'][0][j]
            dataT_tmp['normal'] = dataT['normal'][0][j]
            dataT_tmp['feat'] = dataT['feat'][0][j]
            overlap_val = dataS['overlap'][0][j]
        else:
            # which means we are using the extend data
            dataS_tmp = {}
            dataS_tmp['pc'] = dataS['pc'][j]
            dataS_tmp['normal'] = dataS['normal'][j]
            dataS_tmp['feat'] = dataS['feat'][j]
             
            # target domain data
            dataT_tmp = {}
            dataT_tmp['pc'] = dataT['pc'][j]
            dataT_tmp['normal'] = dataT['normal'][j]
            dataT_tmp['feat'] = dataT['feat'][j]

            overlap_val = dataS['overlap'][j]


        #if 'matterport' in args.dataList:
        if 0:
            overlap = '0-0.1' if overlap_val <= 0.1 else '0.1-0.5' if overlap_val <= 0.5 else '0.5-1.0'
        else:
            overlap = '0-0.1' if overlap_val <= 0.1 else '0.1-1.0'



        # matching 
        if args.hybrid == 0:
            R_hat = Spectral_Matching_M(dataS_tmp, dataT_tmp, args.fitmethod, para)
        else:
            # Hybrid Representation
            para.hybrid_method = args.hybrid_method
            dataS_dict = {}
            dataT_dict = {}

            if '360' in para.hybrid_method:
                dataS_dict['360'] = dataS_tmp
                dataT_dict['360'] = dataT_tmp
            if 'plane' in para.hybrid_method:
                dataS_tmp = {}
                dataT_tmp = {}
                # get plane data

                gt_src, pred_src, gt_tgt, pred_tgt = util.process_plane_point(dataS['path'][j], dataT['path'][j], args.dataList)
                if np.sum(pred_src) == 0:

                    no_plane += 1

                    continue

                pred_src_n, pred_tgt_n = util.normal_plane_point(pred_src[:,3:], pred_tgt[:,3:])
                if 'matterport' in args.dataList:
                    pass
                else:
                    gt_src_n,  gt_tgt_n = util.normal_plane_point(gt_src[:,3:], gt_tgt[:,3:])

                dataS_tmp['pc'] = pred_src[:,:3]
                dataT_tmp['pc'] = pred_tgt[:,:3]
                dataS_tmp['normal'] = pred_src_n
                dataT_tmp['normal'] = pred_tgt_n

                dataS_dict['plane'] = dataS_tmp
                dataT_dict['plane'] = dataT_tmp

                para.w_pair['plane'] = np.sqrt(args.w_plane_2)
                para.w_plane = args.w_plane_1

            if 'topdown' in para.hybrid_method:
                dataS_tmp = {}
                dataT_tmp = {}
                topdown_data = util.process_topdown_mat(dataS['path'][j], dataT['path'][j], args.dataList)
                if topdown_data == 0:
                    no_topdown += 1
                    continue
                #import
                dataS_tmp['feat'] = topdown_data['feat_s']
                dataT_tmp['feat'] = topdown_data['feat_t']
                dataS_tmp['normal'] = topdown_data['nor_s']
                dataT_tmp['normal'] = topdown_data['nor_t']
                dataS_tmp['pc'] = topdown_data['pos_s']
                dataT_tmp['pc'] = topdown_data['pos_t']
                #dataS_tmp['Rst'] = topdown_data['R_s2t']
                dataS_dict['topdown'] = dataS_tmp
                dataT_dict['topdown'] = dataT_tmp

                para.w_topdown = args.w_topdown
            


            if args.hybrid == 1:
                
                #if overlap_val >0.5:
                #import pdb; pdb.set_trace()
                if args.save_primitive == 1:
                    if overlap ==  '0.1-0.5':
                        dataDict = {'dataS_dict': dataS_dict, 'dataT_dict': dataT_dict, 'R_gt_44': R_gt_44}
                        primitives.append(dataDict)

                if args.draw_corres == 0:
                    R_hat, overlap_val = Hybrid_Spectral_Matching_M(dataS_dict, dataT_dict, args.fitmethod, R_gt_44, para.hybrid_method, para)
                else:
                    R_hat, overlap_val, sourcePC_list, targetPC_list, return_corres, points_num, points_tgt_num = Hybrid_Spectral_Matching_M(dataS_dict, dataT_dict, args.fitmethod, R_gt_44, para.hybrid_method, para)


            elif args.hybrid == 2:
                R_hat = Hybrid_Spectral_Matching(dataS_dict, dataT_dict, args.fitmethod, para)
            elif args.hybrid == 3:
                # just for finding good example for paper showing
                R_hat, overlap_val = Hybrid_Spectral_Matching_M(dataS_dict, dataT_dict, args.fitmethod, R_gt_44, para.hybrid_method, para)
                R_hat_360, _ = Hybrid_Spectral_Matching_M(dataS_dict, dataT_dict, 'sm_v2', R_gt_44, '360', para)

        # average speed
        time_this = time.time()-st
        speedBenchmark.append(time_this)
            
        # compute rotation error and translation error

        if isinstance(R_hat, list):
            
            if 1:
                ad_min = 360
                t_tmp = R_hat[0][:3,3]
                tmp_idx = 0
                for k in range(len(R_hat)):
                    ad_tmp = util.angular_distance_np(R_hat[k][:3,:3].reshape(1,3,3),R_gt[np.newaxis,:,:])[0]
                    if ad_tmp < ad_min:
                        ad_min = ad_tmp
                        t_tmp = R_hat[k][:3,3]
                        tmp_idx = k
            else:
                tmp_idx = np.where(overlap_val==np.max(overlap_val))[0][0]
                #import pdb; pdb.set_trace()
                ad_min = util.angular_distance_np(R_hat[tmp_idx][:3,:3].reshape(1,3,3),R_gt[np.newaxis,:,:])[0]
                t_tmp = R_hat[tmp_idx][:3,3]
            
            if args.numMatches == 5:
                best_distribution[tmp_idx] += 1

            ad_this = ad_min

            ad_blind_this = util.angular_distance_np(R_gt[np.newaxis,:,:],np.eye(3)[np.newaxis,:,:])[0]
            translation_this = np.linalg.norm(np.matmul((R_hat[tmp_idx][:3,:3] - R_gt_44[:3,:3]),dataS_tmp['pc'].mean(0).reshape(3)) + t_tmp - R_gt_44[:3,3])
            translation_this = np.linalg.norm(t_tmp - R_gt_44[:3,3])
            translation_blind_this = np.linalg.norm(t_tmp - R_gt_44[:3,3])

            # save result for this pair
            R_pred_44=np.eye(4)
            R_pred_44[:3,:3]=R_hat[tmp_idx][:3,:3]
            R_pred_44[:3,3]=t_tmp    
            
            if args.hybrid == 3:
                ad_360 = util.angular_distance_np(R_hat_360[0][:3,:3].reshape(1,3,3),R_gt[np.newaxis,:,:])[0]
                if (ad_360 - ad_this > 15) and ad_this < 10:
                    print("Good!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    scene_id = dataS['path'][j][0].split('/')[-2]
                    scan_s_id = dataS['path'][j][0].split('/')[-1]
                    scan_t_id = dataT['path'][j][0].split('/')[-1]
                    print('-'.join([scene_id,scan_s_id,scan_t_id]))
                    #good_id.append('-'.join([scene_id,scan_s_id,scan_t_id]))
        else:
            t_hat = R_hat[:3,3]
            R_hat = R_hat[:3,:3]
            ad_this = util.angular_distance_np(R_hat, R_gt[np.newaxis,:,:])[0]
            ad_blind_this = util.angular_distance_np(R_gt[np.newaxis,:,:],np.eye(3)[np.newaxis,:,:])[0]
            #translation_this = np.linalg.norm(np.matmul((R_hat - R_gt_44[:3,:3]),dataS_tmp['pc'].mean(0).reshape(3)) + t_hat - R_gt_44[:3,3])
            translation_this = np.linalg.norm(t_tmp - R_gt_44[:3,3])
            translation_blind_this = np.linalg.norm(t_hat - R_gt_44[:3,3])

            # save result for this pair
            R_pred_44=np.eye(4)
            R_pred_44[:3,:3]=R_hat
            R_pred_44[:3,3]=t_hat
        #import pdb; pdb.set_trace() 
        #if args.draw_corres and ad_this < 5:
        if args.draw_corres:
            print("ad error:", ad_this)
            #import pdb; pdb.set_trace()
            tmp = util.draw_correspondence(sourcePC_list, targetPC_list, return_corres, dataS['path'][j][0], dataT['path'][j][0], dataS['R'][j], dataT['R'][j], points_num, points_tgt_num, center_s.mean(0))

        if args.print_each:

            scene_id = dataS['path'][j][0].split('/')[-2]
            scan_s_id = dataS['path'][j][0].split('/')[-1]
            scan_t_id = dataT['path'][j][0].split('/')[-1]

        
        error_stats.append({'err_ad':ad_this,
            'err_t':translation_this,'err_blind':ad_blind_this,'err_t_blind':translation_blind_this,'overlap':overlap_val, 'R_gt':R_gt_44,'R_pred_44':R_pred_44})
            
        # update statics
        adstatsOverlaps[overlap].append(ad_this)
        transstatsOverlaps[overlap].append(translation_this)

        # print log
        log.info(f"average processing time per pair: {np.sum(speedBenchmark)/len(speedBenchmark)}")
        log.info(f"R_hat:{R_hat}")
        log.info(f"ad/ad_blind this :{ad_this}/{ad_blind_this}\n")

        # print progress bar
        Bar.suffix = '{dataset:10}: [{0:3}/{1:3}] | Total: {total:} | ETA: {eta:}'.format(j, num_data, total=bar.elapsed_td, eta=bar.eta_td,dataset=dataset_name)
        bar.next()

        if (j+1) % 20 == 0:
        #if 1:
            np.save(f"{exp_dir}/{args.exp}.result.npy",error_stats)
            sss=''
            total_ad = 0
            total_tran = 0
            total_num = 0
            for overlap in Overlaps:
                
                if args.detect_each:
                    if overlap == '0-0.1':
                        pre_len = non_overlap_len
                        non_overlap_len = len(adstatsOverlaps[overlap])
                        sss += f"rotation, overlap:{overlap},nobs:{len(adstatsOverlaps[overlap][pre_len:non_overlap_len])}, mean:{np.mean(adstatsOverlaps[overlap][pre_len:non_overlap_len])} "
                    elif overlap == '0.1-0.5':
                        pre_len = small_overlap_len
                        small_overlap_len = len(adstatsOverlaps[overlap])
                        sss += f"rotation, overlap:{overlap},nobs:{len(adstatsOverlaps[overlap][pre_len:small_overlap_len])}, mean:{np.mean(adstatsOverlaps[overlap][pre_len:small_overlap_len])} "
                    elif overlap == '0.5-1.0':
                        pre_len = large_overlap_len
                        large_overlap_len = len(adstatsOverlaps[overlap])
                        sss += f"rotation, overlap:{overlap},nobs:{len(adstatsOverlaps[overlap][pre_len:large_overlap_len])}, mean:{np.mean(adstatsOverlaps[overlap][pre_len:large_overlap_len])} "
                else:
                    if overlap == '0-0.1':
                        pre_len = non_overlap_len
                        non_overlap_len = len(adstatsOverlaps[overlap])
                    sss += f"rotation, overlap:{overlap},nobs:{len(adstatsOverlaps[overlap])}, mean:{np.mean(adstatsOverlaps[overlap])} "
                
                total_ad += np.sum(adstatsOverlaps[overlap])
                total_num += len(adstatsOverlaps[overlap])
                
            if args.filename is not None:
                print(sss, file=f)
            else:
                print(sss)
            sss=''
            for overlap in Overlaps:
                sss += f"translation, overlap:{overlap},nobs:{len(transstatsOverlaps[overlap])}, mean:{np.mean(transstatsOverlaps[overlap])} "
                total_tran += np.sum(transstatsOverlaps[overlap])
            
            if args.filename is not None:

                print(sss, file=f)
                print("rotation mean:", total_ad / total_num, file=f)
                print("translation mean:", total_tran / total_num, file=f)
                print("\n", file=f)
            else:
                print(sss)
                print("rotation mean:", total_ad / total_num)
                print("translation mean:", total_tran / total_num)
               
        if j == args.maxIter:
            print()
            break
    
    if args.numMatches == 5:
        print(best_distribution)
    if args.save_primitive:
        np.save(primitive_file, primitives)
        import pdb; pdb.set_trace()



