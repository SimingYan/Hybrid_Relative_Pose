import open3d as o3d
#import matlab.engine
#eng = matlab.engine.start_matlab()
#eng.addpath("./optimization/")
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import train_op, torch_op
from utils.torch_op import v,npy
from utils.log import AverageMeter
from utils import log
import config
from tensorboardX import SummaryWriter
import cv2
import util
import time
import re
import glob
from opts import opts
from utils.dotdict import *
from utils.factory import trainer
from model.mymodel import RelationNet as MODEL
from utils.callbacks import PeriodicCallback, OnceCallback, ScheduledCallback,CallbackLoc
import copy
import hungarian
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from optimization import local_utils
from optimization.local_utils import run_optimization, objective
import scipy.io as sio

def buildDataset(args):
    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    if 'suncg' in args.dataList:
        from datasets.SUNCG_lmdb import SUNCG_lmdb as Dataset
    elif 'scannet' in args.dataList:
        from datasets.ScanNet_lmdb import ScanNet_lmdb as Dataset
    elif 'matterport' in args.dataList:
        from datasets.Matterport3D import Matterport3D as Dataset
    else:
        raise Exception("unknown dataset!")

    shuffle = False
    val_dataset = Dataset('test', nViews=args.nViews,meta=False,rotate=False,rgbd=True,hmap=False,segm=False,normal=True,\
        list_=f"./data/dataList/{args.dataList}.npy",singleView=args.single_view,denseCorres=args.featurelearning,reproj=(args.reproj==1),\
        representation=args.representation,dynamicWeighting=args.dynamicWeighting,snumclass=args.snumclass,corner=False,local=True,eval_local=args.eval_local,
        local_eval_list=args.local_eval_list)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers,drop_last=False,collate_fn=util.collate_fn_cat, worker_init_fn=worker_init_fn)

    return val_loader

def resume_checkpoint(net, net_path):
    checkpoint = torch.load(net_path)
    state_dict = checkpoint['state_dict']
    net.load_state_dict(state_dict)
    print('resume network weights from {0} successfully'.format(net_path))


def log_string(content, fp):
    fp.write(content + '\n')
    print(content)

def eval_fn(eval_dict, fp):
    fh = open(fp, 'w')
    results = []
    for key in eval_dict:
        Kth = int(key.split('-')[-1])
        R_g = eval_dict[key]['global']
        R_l = eval_dict[key]['local']
        R_gt = eval_dict[key]['gt']
        err_r_g = util.angular_distance_np(R_g[:3,:3], R_gt[:3,:3])
        err_r_l = util.angular_distance_np(R_l[:3,:3], R_gt[:3,:3])
        err_t_g = np.linalg.norm(R_g[:3,3]- R_gt[:3,3])
        err_t_l = np.linalg.norm(R_l[:3,3]- R_gt[:3,3])
        results.append({'err_r_g': err_r_g, 'err_r_l':err_r_l,
                    'err_t_g': err_t_g, 'err_t_l':err_t_l,
                    'overlap': eval_dict[key]['overlap'],'Kth':Kth})
    
    for k in range(5):
        if k>0: break
        # stats for global 
        results_this = [res for res in results if res['Kth']==k]
        err_r_g = [res['err_r_g'] for res in results_this]
        err_t_g = [res['err_t_g'] for res in results_this]
        log_string('total # %d,  global module error rotation: %.5f, error translation: %.5f' % (len(err_r_g), np.mean(err_r_g), np.mean(err_t_g)), fh)
        err_r_g = [res['err_r_g'] for res in results_this if res['overlap'] < 0.1]
        err_t_g = [res['err_t_g'] for res in results_this if res['overlap'] < 0.1]
        log_string('\ttotal # %d,  small overlap: rotation: %.5f, error translation: %.5f' % (len(err_r_g), np.mean(err_r_g), np.mean(err_t_g)), fh)
        err_r_g = [res['err_r_g'] for res in results_this if res['overlap'] >= 0.1]
        err_t_g = [res['err_t_g'] for res in results_this if res['overlap'] >= 0.1]
        log_string('\ttotal # %d,  large overlap: rotation: %.5f, error translation: %.5f' % (len(err_r_g), np.mean(err_r_g), np.mean(err_t_g)), fh)

        # stats for local 
        err_r_l = [res['err_r_l'] for res in results_this]
        err_t_l = [res['err_t_l'] for res in results_this]
        log_string('total # %d,  local module error rotation: %.5f, error translation: %.5f' % (len(err_r_l), np.mean(err_r_l), np.mean(err_t_l)), fh)
        err_r_l = [res['err_r_l'] for res in results_this if res['overlap'] < 0.1]
        err_t_l = [res['err_t_l'] for res in results_this if res['overlap'] < 0.1]
        log_string('\ttotal # %d,  small overlap: rotation: %.5f, error translation: %.5f' % (len(err_r_l), np.mean(err_r_l), np.mean(err_t_l)), fh)
        err_r_l = [res['err_r_l'] for res in results_this if res['overlap'] >= 0.1]
        err_t_l = [res['err_t_l'] for res in results_this if res['overlap'] >= 0.1]
        log_string('\ttotal # %d,  large overlap: rotation: %.5f, error translation: %.5f' % (len(err_r_l), np.mean(err_r_l), np.mean(err_t_l)), fh)
    fh.close()


if __name__ == '__main__':

    # parse parameters 
    opt = opts()
    opt.parser.add_argument('--method', type=str, default='ours', help='ours,4pcs,gr') 
    opt.parser.add_argument('--dataset', type=str, default='suncg', help='suncg,scannet,matterport') 

    args = opt.parse()

    # define network 
    if 'scannet' in args.dataList:
        num_s = 21
    elif 'matterport' in args.dataList:
        num_s = 21
    elif 'suncg' in args.dataList:
        num_s = 16

    # load global module results, create data list for local module's data loader
    postfix = '' if args.method == 'ours' else args.method
    if args.method == 'ours':
      dataS=sio.loadmat('./data/test_data/%s_source.mat' % args.dataset)
      dataT=sio.loadmat('./data/test_data/%s_target.mat' % args.dataset)
    else:
      dataS=sio.loadmat('./data/test_data/%s_source_%s.mat' % (args.dataset, args.method))
      dataT=sio.loadmat('./data/test_data/%s_target_%s.mat' % (args.dataset, args.method))

    newList = [] 
    err_r=[]
    for i in range(len(dataS['path'])):
        tp = dataS['path'][i][0]
        tp1 = dataT['path'][i][0]
        if args.dataset=='suncg' or args.dataset=='matterport':
            sceneID = tp.split('/')[-3] + '-' +  tp.split('/')[-2]
        else:
            sceneID = tp.split('/')[-2]
        
        id_src = int(tp.split('/')[-1])
        id_tgt = int(tp1.split('/')[-1])
        key = '%s-%06d-%06d' % (sceneID, id_src, id_tgt)
        newList.append([sceneID, tp.split('/')[-1], tp1.split('/')[-1], dataS['pred_pose'][i], dataS['gt_pose'][i], dataS['overlap'][0, i]])
        err_r.append(util.angular_distance_np(dataS['pred_pose'][i][0,:3,:3], dataS['gt_pose'][i][:3,:3]))

        out_dir = './data/dataList/%s_local/' % args.dataset
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save('%s/release_eval.npy' % out_dir, newList)
    print(np.mean(err_r))
    netG=MODEL(input_chal=10,num_s=num_s).cuda()

    # resume checkpoint
    resume_checkpoint(netG, args.model)
    
    
    
    # build data loader
    val_loader=buildDataset(args)

    # iterate through all data 
    eval_dict = {}
    confusion_matrix_data = {'pred':[], 'gt':[]}
    fp = './tmp/%s_local_result_%.3f_%.3f_%.3f.txt' % (args.dataset, args.thre_coplane, args.thre_parallel, args.thre_perp)
    
    with torch.set_grad_enabled(False):
        for batch_id, data in enumerate(val_loader):
            if 'suncg' in args.dataList :
                THRESH = THRESH = np.array([args.thre_coplane,args.thre_parallel,args.thre_perp]) 
            elif 'scannet' in args.dataList:
                THRESH = np.array([args.thre_coplane,args.thre_parallel,args.thre_perp])
            elif 'matterport' in args.dataList:
                THRESH = np.array([args.thre_coplane,args.thre_parallel,args.thre_perp])
            rgb,depth,dataMask,R,overlap = v(data['rgb']),v(data['depth']),v(data['dataMask']),v(data['R']),npy(data['overlap'])
            
            pointcloud = v(data['pointcloud'])
            
            igt = v(data['igt'])
            if args.local_method == 'patch':
              pair = v(data['pair'])
              rel_dst = v(data['rel_dst'])
              rel_cls = v(data['rel_cls'])
              rel_ndot = v(data['rel_ndot'])
              rel_valid = v(data['rel_valid'])
              plane_idx = v(data['plane_idx'])
              plane_center = v(data['plane_center'])
            elif args.local_method == 'point':
              rel_cls_pts = v(data['rel_cls_pts'])
              pair_pts = v(data['pair_pts'])
              imgPCgrid = v(data['uv_pts']).long()
              normdot2 = v(data['normdot2'])
              dst2 = v(data['dst2'])
            n = R.shape[0]
            semantic = v(data['semantic'])
            pointcloud_semantic = torch.cat((pointcloud[:, 0, 9, :], pointcloud[:, 1, 9, :]), -1)
            pointcloud_s = pointcloud[:, 0,0:3, ...]
            pointcloud_t = pointcloud[:, 1,0:3, ...]
            pointcloud_n_s = pointcloud[:, 0,3:6, ...]
            pointcloud_n_t = pointcloud[:, 1,3:6, ...]
            pointcloud_c_s = pointcloud[:, 0,6:9, ...]
            pointcloud_c_t = pointcloud[:, 1,6:9, ...]
            PerspectiveValidMask = data['PerspectiveValidMask'].float().cuda()
            R = igt

            b, c, n = pointcloud_t.shape
            rot_v = torch.autograd.Variable(torch.eye(4).unsqueeze(0).repeat(b,1,1).float().cuda(),requires_grad=True)
            
            
            pointcloud_t_cat = torch.cat((pointcloud_t, pointcloud_n_t,pointcloud_c_t, v(np.zeros([b, 1, n]))), 1)
            pointcloud_s_cat = torch.cat((pointcloud_s, pointcloud_n_s,pointcloud_c_s,v(np.ones([b, 1, n]))), 1)
            
            if 'suncg' in args.dataList or 'matterport' in args.dataList:
              semantic_cat = torch.cat((semantic[:,0,:,160:160*2],semantic[:,1,:,160:160*2]),0)
              rgb_cat = torch.cat((rgb[:,0,:,:,160:160*2],rgb[:,1,:,:,160:160*2]))
              PerspectiveValidMask_cat = torch.cat((PerspectiveValidMask[:,0,:,:,160:160*2],PerspectiveValidMask[:,0,:,:,160:160*2]))
            elif 'scannet' in args.dataList:
              semantic_cat = torch.cat((semantic[:,0,100-48:100+48,200-64:200+64],semantic[:,1,100-48:100+48,200-64:200+64]),0)
              rgb_cat = torch.cat((rgb[:,0,:,100-48:100+48,200-64:200+64],rgb[:,1,:,100-48:100+48,200-64:200+64]))
              PerspectiveValidMask_cat = torch.cat((PerspectiveValidMask[:,0,:,100-48:100+48,200-64:200+64],
                                                PerspectiveValidMask[:,1,:,100-48:100+48,200-64:200+64]))

            # mask the pano
            dataMask = dataMask[:,0,:,:,:]
            
            pointcloud = torch.cat((pointcloud_s_cat, pointcloud_t_cat), 2)
            
            
            pred_cls,pointwise_logits,img_logits      = netG(pointcloud,pair_pts, rgb_cat, imgPCgrid)
            
            pred_cls_reduce = torch.argmax(pred_cls, 2)
            
            print_confusion = False
            if print_confusion:
                if self.global_step%10==0:
                    conf = np.zeros([4,4])
                    if len(self.confusion_matrix_data['pred']):
                        pred = np.concatenate(self.confusion_matrix_data['pred'])
                        gt = np.concatenate(self.confusion_matrix_data['gt'])
                        for jj in range(4):
                            for kk in range(4):
                                conf[jj,kk] = sum((gt==jj) & (pred==kk))
                        conf = conf / (conf.sum(1, keepdims=True)+1e-16)
                        print(conf)

            if args.eval_local:

              err_angle = []
              err_t = []
              err_angle_init = []
              err_t_init = []
              num_point = pointcloud.shape[-1]
              

              for i in range(b):

                print(data['imgsPath'][0][i], data['imgsPath'][1][i])

                data_s_nn = {'pos':np.concatenate((npy(pointcloud[i, :3, :num_point//2]).T,
                                data['pos_s_360'][i])),
                            'nor':np.concatenate((npy(pointcloud[i, 3:6, :num_point//2]).T,
                                data['nor_s_360'][i]))}
                data_t_nn = {'pos':np.concatenate((npy(pointcloud[i, :3, num_point//2:]).T,
                                data['pos_t_360'][i])),
                            'nor':np.concatenate((npy(pointcloud[i, 3:6, num_point//2:]).T,
                                data['nor_t_360'][i]))}
                
                if args.local_method == 'point':
                  pred_prob_this = npy(F.softmax(pred_cls[i], -1))
                  pred_this = np.zeros([len(pred_prob_this)])
                  
                  tp = (pred_prob_this[:,1:] > THRESH[None, :])
                  pred_this[tp.sum(1)==1] = np.argmax(tp,1)[tp.sum(1)==1]+1
                 
                
                  if print_confusion:
                    confusion_matrix_data['pred'].append(pred_this)
                    confusion_matrix_data['gt'].append(npy(rel_cls_pts[i]))

                  ind = np.where(pred_this!=0)[0]
                  rel_this = pred_this[ind]
                  # rel_this = npy(rel_cls_pts[i])[ind]
                  
                  print('num perp:%d, num parallel: %d, num coplane: %d' % ((rel_this==1).sum(), (rel_this==2).sum(), (rel_this==3).sum()))
                  gt = npy(rel_cls_pts[i])[ind]
                  prec1 = np.mean(rel_this[rel_this==1] == gt[rel_this==1])
                  prec2 = np.mean(rel_this[rel_this==2] == gt[rel_this==2])
                  prec3 = np.mean(rel_this[rel_this==3] == gt[rel_this==3])
                  print('precision perp:%.3f, parallel: %.3f, coplane: %.3f' % (prec1, prec2, prec3))

                  pairID_s = npy(pair_pts[i, ind, 0]).astype('int')
                  pairID_t = npy(pair_pts[i, ind, 1]).astype('int')
                  data_s = {'pos':npy(pointcloud[i, :3, :num_point//2]).T,
                            'nor':npy(pointcloud[i, 3:6, :num_point//2]).T}
                  data_t = {'pos':npy(pointcloud[i, :3, num_point//2:]).T,
                            'nor':npy(pointcloud[i, 3:6, num_point//2:]).T}
                
                print(int(data['eval_key'][i].split('-')[-1]))

                if 'suncg' in args.dataList or 'matterport' in args.dataList:
                    sceneID = data['eval_key'][i].split('-')[0] + '-' + data['eval_key'][i].split('-')[1]
                    id_src = int(data['eval_key'][i].split('-')[2])
                    id_tgt = int(data['eval_key'][i].split('-')[3])
                    Kth = int(data['eval_key'][i].split('-')[4])
                else:
                    sceneID = data['eval_key'][i].split('-')[0]
                    id_src = int(data['eval_key'][i].split('-')[1])
                    id_tgt = int(data['eval_key'][i].split('-')[2])
                    Kth = int(data['eval_key'][i].split('-')[3])
                if Kth != 0:
                    continue
                R_pred, t_pred, loss_local, loss_local_init,loss_p2plane, loss_p2plane_init, stats, valid = \
                  local_utils.solve(data_s_nn, data_t_nn, data_s, data_t, pairID_s, pairID_t, rel_this, npy(igt[i])[:3,:3],npy(igt[i][:3,3]))
                R_pred_44 = np.eye(4)
                R_pred_44[:3,:3]=R_pred
                R_pred_44[:3,3] = t_pred
                
                eval_dict[data['eval_key'][i]] = {'global':npy(data['pred_pose'][i]),
                                        'local':np.matmul(R_pred_44, npy(data['pred_pose'][i])),
                                        'loss_local':loss_local,
                                        'loss_local_init':loss_local_init,
                                        'gt':npy(data['gt_pose'][i]),
                                        'overlap':overlap[i]}
            if batch_id % 5 == 0:
                eval_fn(eval_dict, fp)

        # run evaluation
        eval_fn(eval_dict, fp)
