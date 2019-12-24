# Hybrid version of relative pose estimation
from open3d import *
import glob
import numpy as np
import cv2
import sys
from utils import torch_op
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
#from RPModule.hybrid_util import *
from sklearn.neighbors import NearestNeighbors

import torch
import util
import copy
import logging
from sklearn.cluster import KMeans
from util import angular_distance_np
import scipy.io as sio

from RPModule.rputil import opts, greedy_rounding
from RPModule.rputil import *

# base method to calculate relative pose
def horn87_np(src, tgt,weight=None):
    '''
    # src: [(k), 3, n]
    # tgt: [(k), 3, n]
    # weight: [(k), n]
    # return: 
    # R: [(k),3,3]
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
    # calculate the quaternion and get rotation matrix
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

# use single relative pose
def fit_spectral(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,mu,row_dict,col_dict,num_s_dict,num_t_dict, hybrid_method):
    """
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    row:   1st component of correspondence pair
    col:   2nd component of correspondence pair
    numFea_s: number of source keypoint
    numFea_t: number of target keypoint
    """
    num_alter = 1
    offset    = 50
    EPS       = 1e-12
    
    #import pdb; pdb.set_trace()
    matrix_size = 0 
    row_total = []
    col_total = []
    allSP_total = []
    allTP_total = []
    allSN_total = []
    allTN_total = []
    a_i1i2j1j2_total = []
    w_i1i2j1j2_total = []

    if '360' in hybrid_method:
        # compute center
        SPmean = (allSP_dict['360']*allWP_dict['360'][:,np.newaxis]).sum(0)/(allWP_dict['360'].sum()+EPS)
        allSPc = allSP_dict['360'] - SPmean
        TPmean = (allTP_dict['360']*allWP_dict['360'][:,np.newaxis]).sum(0)/(allWP_dict['360'].sum()+EPS)
        allTPc = allTP_dict['360'] - TPmean

        # use 360 representation to initialize R,t
        allS = np.concatenate((allSPc,allSN_dict['360']))
        allT = np.concatenate((allTPc,allTN_dict['360']))
        allW = np.concatenate((allWP_dict['360']*mu,allWN_dict['360']))
        R_hat = horn87_np(allS.T,allT.T,allW)
        t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

        R_cur = R_hat.squeeze()
        t_cur = t_hat.squeeze()

        # get the entry of matrix A
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN_dict['360'].T)-allTN_dict['360'].T,2).sum(0))
    
        a_i1i2j1j2 = allWP_dict['360']*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
    
        row_total.append(row_dict['360'])
        col_total.append(col_dict['360'])
        a_i1i2j1j2_total.append(a_i1i2j1j2)
        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['360'])
        allSP_total.append(allSP_dict['360'])
        allTP_total.append(allTP_dict['360'])
        allSN_total.append(allSN_dict['360'])
        allTN_total.append(allTN_dict['360'])

        matrix_size += num_s_dict['360'] * num_t_dict['360']

    # additional hybrid represenation
    if 'plane' in hybrid_method:

        # calculate the entry of matrix A
        a_i1i2j1j2_p = allWP_dict['plane']
        a_i1i2j1j2_p[a_i1i2j1j2_p<0] = 0
        a_i1i2j1j2_p = a_i1i2j1j2_p.reshape(2, -1).sum(0)
        row_p = row_dict['plane'] + matrix_size
        col_p = col_dict['plane'] + matrix_size

        row_total.append(row_p)
        col_total.append(col_p)

        a_i1i2j1j2_total.append(a_i1i2j1j2_p)

        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['plane'])
        allSP_total.append(allSP_dict['plane'])
        allTP_total.append(allTP_dict['plane'])
        allSN_total.append(allSN_dict['plane'])
        allTN_total.append(allTN_dict['plane'])

        matrix_size += (num_s_dict['plane'] * num_t_dict['plane'])

    if len(a_i1i2j1j2_total) == 1:
        row_total = row_total[0]
        col_total = col_total[0]
        a_i1i2j1j2_total = a_i1i2j1j2_total[0]
        w_i1i2j1j2_total = w_i1i2j1j2_total[0]
        allSP_total = allSP_total[0]
        allTP_total = allTP_total[0]
        allSN_total = allSN_total[0]
        allTN_total = allTN_total[0]

    else:
        row_total = np.concatenate(row_total)
        col_total = np.concatenate(col_total)
        a_i1i2j1j2_total = np.concatenate(a_i1i2j1j2_total)
        w_i1i2j1j2_total = np.concatenate(w_i1i2j1j2_total)
        allSP_total = np.concatenate(allSP_total)
        allTP_total = np.concatenate(allTP_total)     
        allSN_total = np.concatenate(allSN_total)
        allTN_total = np.concatenate(allTN_total)

    # construct the A matrix, compute most significant eigenvector
    A = csc_matrix((a_i1i2j1j2_total, (row_total,col_total)), shape=(matrix_size, matrix_size))
    A = A+A.T
    #import pdb; pdb.set_trace()       
    top_k = 1

    vals, u_ = sparse.linalg.eigs(A, k=top_k)
    u_ = u_[:, top_k-1]
    u_=u_.real
    u_ /= np.linalg.norm(u_)
    x_i1i2j1j2 = (u_[row_total]*u_[col_total]).squeeze()
        
    x_i1i2j1j2[x_i1i2j1j2<0] = 0
    x_i1i2j1j2 *= w_i1i2j1j2_total
        
    # initialize new weight 
    allW = np.tile(x_i1i2j1j2,4)
    # apply mu on weight corresponding to position
    allW[:len(allW)//2]*=mu

    # compute center
    # import pdb; pdb.set_trace()
    allWP = allW[:len(allW)//2]
    SPmean = (allSP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allSPc = allSP_total - SPmean
    TPmean = (allTP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allTPc = allTP_total - TPmean
    #import pdb; pdb.set_trace()

    allS = np.concatenate((allSPc,allSN_total))
    allT = np.concatenate((allTPc,allTN_total))
    # compute R,t using current weight
    R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
    t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
    
    R_cur = R_tp
    t_cur = t_tp

    R        = np.eye(4)
    R[:3,:3] = R_cur
    R[:3,3]  = t_cur

    return R

# use multi-output relative pose
def fit_spectral_v2(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,mu,row_dict,col_dict,num_s_dict,num_t_dict, hybrid_method):
    """
    multi-output relative pose
    allSP: source keypoint position
    allTP: target keypoint position
    allSN: source keypoint normal
    allTN: target keypoint normal
    allWP: point weight
    allWP: weight for position
    allWN: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    row:   1st component of correspondence pair
    col:   2nd component of correspondence pair
    numFea_s: number of source keypoint
    numFea_t: number of target keypoint
    """
    num_alter = 1
    offset    = 50
    EPS       = 1e-12
    
    #import pdb; pdb.set_trace()
    matrix_size = 0 
    row_total = []
    col_total = []
    allSP_total = []
    allTP_total = []
    allSN_total = []
    allTN_total = []
    a_i1i2j1j2_total = []
    w_i1i2j1j2_total = []
    
    # 360-image representation
    if '360' in hybrid_method:
        # compute center
        SPmean = (allSP_dict['360']*allWP_dict['360'][:,np.newaxis]).sum(0)/(allWP_dict['360'].sum()+EPS)
        allSPc = allSP_dict['360'] - SPmean
        TPmean = (allTP_dict['360']*allWP_dict['360'][:,np.newaxis]).sum(0)/(allWP_dict['360'].sum()+EPS)
        allTPc = allTP_dict['360'] - TPmean

        # use 360 representation to initialize R,t
        allS = np.concatenate((allSPc,allSN_dict['360']))
        allT = np.concatenate((allTPc,allTN_dict['360']))
        allW = np.concatenate((allWP_dict['360']*mu,allWN_dict['360']))
        R_hat = horn87_np(allS.T,allT.T,allW)
        t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

        R_cur = R_hat.squeeze()
        t_cur = t_hat.squeeze()

        # get the entry of matrix A
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN_dict['360'].T)-allTN_dict['360'].T,2).sum(0))
    
        a_i1i2j1j2 = allWP_dict['360']*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
        row_360 = row_dict['360'] + matrix_size
        col_360 = col_dict['360'] + matrix_size

        row_total.append(row_360)
        col_total.append(col_360)

        a_i1i2j1j2_total.append(a_i1i2j1j2)

        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['360'])

        allSP_total.append(allSP_dict['360'])
        allTP_total.append(allTP_dict['360'])
        allSN_total.append(allSN_dict['360'])
        allTN_total.append(allTN_dict['360'])

        matrix_size += num_s_dict['360'] * num_t_dict['360']    

    # plane represenation
    if 'plane' in hybrid_method:

        # calculate the entry of matrix A
        a_i1i2j1j2_p = allWP_dict['plane']
        a_i1i2j1j2_p[a_i1i2j1j2_p<0] = 0
        a_i1i2j1j2_p = a_i1i2j1j2_p.reshape(2, -1).sum(0)
        row_p = row_dict['plane'] + matrix_size
        col_p = col_dict['plane'] + matrix_size
        
        row_total.append(row_p)
        col_total.append(col_p)
        
        a_i1i2j1j2_p *= 30
        w_i1i2j1j2_dict['plane'] *= 30

        a_i1i2j1j2_total.append(a_i1i2j1j2_p)

        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['plane'])
        allSP_total.append(allSP_dict['plane'])
        allTP_total.append(allTP_dict['plane'])
        allSN_total.append(allSN_dict['plane'])
        allTN_total.append(allTN_dict['plane'])

        matrix_size += (num_s_dict['plane'] * num_t_dict['plane'])
    
    if len(a_i1i2j1j2_total) == 1:
        # one representation
        row_total = row_total[0]
        col_total = col_total[0]
        a_i1i2j1j2_total = a_i1i2j1j2_total[0]
        w_i1i2j1j2_total = w_i1i2j1j2_total[0]
        allSP_total = allSP_total[0]
        allTP_total = allTP_total[0]
        allSN_total = allSN_total[0]
        allTN_total = allTN_total[0]
    else:
        # more than one represenation
        row_total = np.concatenate(row_total)
        col_total = np.concatenate(col_total)
        a_i1i2j1j2_total = np.concatenate(a_i1i2j1j2_total)
        w_i1i2j1j2_total = np.concatenate(w_i1i2j1j2_total)
        allSP_total = np.concatenate(allSP_total)
        allTP_total = np.concatenate(allTP_total)     
        allSN_total = np.concatenate(allSN_total)
        allTN_total = np.concatenate(allTN_total)


    top_k = 6
    R_hat_l = []
    #import pdb;pdb.set_trace()
    # construct the A matrix, compute most significant eigenvector
    A = csc_matrix((a_i1i2j1j2_total, (row_total,col_total)), shape=(matrix_size, matrix_size))
    A = A+A.T  

    for k_ in range(top_k):
        vals, u_ = sparse.linalg.eigs(A, k=top_k)
        u_ = u_[:, k_]
        u_=u_.real
        u_ /= np.linalg.norm(u_)
        x_i1i2j1j2 = (u_[row_total]*u_[col_total]).squeeze()
            
        x_i1i2j1j2[x_i1i2j1j2<0] = 0
        x_i1i2j1j2 *= w_i1i2j1j2_total
            
        # initialize new weight 
        allW = np.tile(x_i1i2j1j2,4)
        # apply mu on weight corresponding to position
        allW[:len(allW)//2]*=mu

        # compute center
        # import pdb; pdb.set_trace()
        allWP = allW[:len(allW)//2]
        SPmean = (allSP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allSPc = allSP_total - SPmean
        TPmean = (allTP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allTPc = allTP_total - TPmean
        #import pdb; pdb.set_trace()

        allS = np.concatenate((allSPc,allSN_total))
        allT = np.concatenate((allTPc,allTN_total))
        # compute R,t using current weight
        R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
        t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
        
        R_cur = R_tp
        t_cur = t_tp

        R        = np.eye(4)
        R[:3,:3] = R_cur
        R[:3,3]  = t_cur

        R_hat_l.append(R)

    return R_hat_l

# use multi-output relative pose
def fit_irls_sm_v2(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,mu,row_dict,col_dict,num_s_dict,num_t_dict, hybrid_method):
    """
    multi-output relative pose
    allSP_dict: source keypoint position
    allTP_dict: target keypoint position
    allSN_dict: source keypoint normal
    allTN_dict: target keypoint normal
    allWP_dict: weight for position
    allWN_dict: weight for normal
    w_i1i2j1j2: weight for correspondence pair
    mu:   a scalar weight for position
    row_dict:   1st component of correspondence pair's index
    col_dict:   2nd component of correspondence pair's indeex
    num_s_dict: number of source keypoint
    num_t_dict: number of target keypoint
    """
    num_reweighted = 5
    num_alter      = 5
    resSigma       = 1
    offset         = 50
    EPS            = 1e-12

    matrix_size = 0 
    row_total = []
    col_total = []
    allSP_total = []
    allTP_total = []
    allSN_total = []
    allTN_total = []
    allWP_total = []
    allWN_total = []

    a_i1i2j1j2_total = []
    w_i1i2j1j2_total = []
    #import pdb; pdb.set_trace() 
    
    if '360' in hybrid_method:

        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['360'])
        
        # 2 times longer than w_i1i2j1j2 cuz we include the pair of correspondence
        allSP_total.append(allSP_dict['360'])  
        allTP_total.append(allTP_dict['360'])
        allSN_total.append(allSN_dict['360'])
        allTN_total.append(allTN_dict['360'])
        allWP_total.append(allWP_dict['360'])
        allWN_total.append(allWN_dict['360'])
       
        row_360 = row_dict['360'] + matrix_size
        col_360 = col_dict['360'] + matrix_size
        row_total.append(row_360)
        col_total.append(col_360)
        
        matrix_size += num_s_dict['360'] * num_t_dict['360']

    if 'plane' in hybrid_method or 'corner' in hybrid_method:
        
        w_i1i2j1j2_total.append(w_i1i2j1j2_dict['plane'])

        allSP_total.append(allSP_dict['plane'])
        allTP_total.append(allTP_dict['plane'])
        allSN_total.append(allSN_dict['plane'])
        allTN_total.append(allTN_dict['plane'])
        allWP_total.append(allWP_dict['plane'])
        allWN_total.append(allWN_dict['plane'])

        row_p = row_dict['plane'] + matrix_size
        col_p = col_dict['plane'] + matrix_size        
        row_total.append(row_p)
        col_total.append(col_p)

        matrix_size += (num_s_dict['plane'] * num_t_dict['plane'])

    if len(allSP_total) == 1:
        # one representation
        row_total = row_total[0]
        col_total = col_total[0]
        w_i1i2j1j2_total = w_i1i2j1j2_total[0]
        allSP_total = allSP_total[0]
        allTP_total = allTP_total[0]
        allSN_total = allSN_total[0]
        allTN_total = allTN_total[0]
        allWP_total = allWP_total[0]
        allWN_total = allWN_total[0]

    else:
        # more than one represenation
        row_total = np.concatenate(row_total)
        col_total = np.concatenate(col_total)
        w_i1i2j1j2_total = np.concatenate(w_i1i2j1j2_total)
        allSP_total = np.concatenate(allSP_total)
        allTP_total = np.concatenate(allTP_total)     
        allSN_total = np.concatenate(allSN_total)
        allTN_total = np.concatenate(allTN_total)
        allWP_total = np.concatenate(allWP_total)
        allWN_total = np.concatenate(allWN_total)


    # initialize R,t
    for j in range(num_reweighted):

        # compute center
        SPmean = (allSP_total*allWP_total[:,np.newaxis]).sum(0)/(allWP_total.sum()+EPS)
        allSPc = allSP_total - SPmean
        TPmean = (allTP_total*allWP_total[:,np.newaxis]).sum(0)/(allWP_total.sum()+EPS)
        allTPc = allTP_total - TPmean

        allS = np.concatenate((allSPc,allSN_total))
        allT = np.concatenate((allTPc,allTN_total))
        allW = np.concatenate((allWP_total*mu,allWN_total))

        # compute R,t using current weight
        R_hat = horn87_np(allS.T,allT.T,allW)
        t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
        # compute new weight
        residualPc = mu*np.power(np.matmul(R_hat.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
        residualN = np.power(np.matmul(R_hat.squeeze(),allSN_total.T)-allTN_total.T,2).sum(0)
        residual = np.concatenate((residualPc,residualN))
        allW = allW * resSigma**2/(resSigma**2+residual)

    R_cur = R_hat.squeeze()
    t_cur = t_hat.squeeze()
    
    R_hat_l = []
    top_k = 1
    flag = 1
    
    # alternate between spectral method and irls
    for j in range(num_alter):
        if j > 0:
            flag = 0
        
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN_total.T)-allTN_total.T,2).sum(0))
    
        a_i1i2j1j2 = np.tile(w_i1i2j1j2_total,2)*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
    
        # construct the A matrix, compute most significant eigenvector
        A = csc_matrix((a_i1i2j1j2, (row_total,col_total)), shape=(matrix_size, matrix_size))
        A = A+A.T
    


        for k_ in range(top_k):
            vals, u_ = sparse.linalg.eigs(A, k=top_k)
            u_ = u_[:,k_]
            u_=u_.real
            u_ /= np.linalg.norm(u_)

            x_i1i2j1j2 = (u_[row_total]*u_[col_total]).squeeze()

            x_i1i2j1j2[x_i1i2j1j2<0] = 0
            x_i1i2j1j2 *= w_i1i2j1j2_total

            # get new weight
            allW = np.tile(x_i1i2j1j2,4)
            # apply mu on weight corresponding to position
            allW[:len(allW)//2]*=mu
        
            for l in range(num_reweighted):
                # get the weight for position
                allWP=allW[:len(allW)//2]
                # compute center
                SPmean = (allSP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
                allSPc = allSP_total - SPmean
                TPmean = (allTP_total*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
                allTPc = allTP_total - TPmean

                allS = np.concatenate((allSPc,allSN_total))
                allT = np.concatenate((allTPc,allTN_total))

                # compute R,t using current weight
                R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
                t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

                # compute new weight
                residualPc = mu*np.power(np.matmul(R_tp.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
                residualN = np.power(np.matmul(R_tp.squeeze(),allSN_total.T)-allTN_total.T,2).sum(0)
                residual = np.concatenate((residualPc,residualN))
                allW = allW * resSigma**2/(resSigma**2+residual)

            R_cur = R_tp
            t_cur = t_tp
            
            R        = np.eye(4)
            R[:3,:3] = R_cur
            R[:3,3]  = t_cur
            
            if flag:
                R_hat_l.append(R)
            else:
                R_hat_l[k_] = R

    return R_hat_l

def Hybrid_Spectral_Matching(dataS, dataT, spectral_method, para):

    allSP_dict = {}
    allTP_dict = {}
    allSN_dict = {}
    allTN_dict = {}
    allWP_dict = {}
    allWN_dict = {}
    w_i1i2j1j2_dict = {}
    row_dict = {}
    col_dict = {}
    num_s_dict = {}
    num_t_dict = {}

    if '360' in para.hybrid_method: 

        sourcePC = dataS['360']['pc']
        targetPC = dataT['360']['pc']

        sourceNormal = dataS['360']['normal']
        targetNormal = dataT['360']['normal']

        sourceFeat = dataS['360']['feat'] / 100.
        targetFeat = dataT['360']['feat'] / 100.
        
        num_s = sourcePC.shape[0] # number of source keypoints
        num_t = targetPC.shape[0] # number of target keypoints

        if num_s < 3 or num_t < 3:
            print("Too few keypoints!")
            return np.eye(4)

        # compute wij based on descriptor distance. 
        pcWij = np.ones((num_s, num_t))
        dij = np.power(np.expand_dims(sourceFeat,1)-np.expand_dims(targetFeat,0),2).sum(2) # distance between features of keypoints
        sigmaij = np.ones(pcWij.shape)*para.sigmaFeat # weight of feature matrix
        wij = np.exp(np.divide(-dij, 2*np.power(sigmaij/5, 2))) # smaller means worse
        nm = np.linalg.norm(wij, axis=1, keepdims=True) # normalization
        equalzero = (nm==0)
        nm[equalzero] = 1
        wij/=nm
        wij[equalzero.squeeze(),:]=0



        # prune for top K correspondence for simplicity
        topK = min(para.topK, wij.shape[1]-1)
        topIdx = np.argpartition(-wij,topK,axis=1)[:, :topK]
        
        corres = np.zeros([2, num_s * topK])
        corres[0, :] = np.arange(num_s).repeat(topK)
        corres[1, :] = topIdx.flatten()
        corres = corres.astype('int')
        num_corres = corres.shape[1]

        if num_corres < 3:
            print("Too few correspondences!")
            return np.eye(4)

        # get the pair of correspondence index
        idx = np.tile(np.arange(num_corres),num_corres).reshape(-1,num_corres)
        idy = idx.T
        valid = (idx > idy)
        idx = idx[valid] # the first index of correspondence pair 
        idy = idy[valid] # the second index of correspondence pair

        # distance consistency
        pci1 = sourcePC[corres[0,idy],:]
        pcj1 = targetPC[corres[1,idy],:]
        pci2 = sourcePC[corres[0,idx],:]
        pcj2 = targetPC[corres[1,idx],:]

        ni1 = sourceNormal[corres[0,idy],:]
        nj1 = targetNormal[corres[1,idy],:]
        ni2 = sourceNormal[corres[0,idx],:]
        nj2 = targetNormal[corres[1,idx],:]

        dis_s = np.linalg.norm(pci1 - pci2,axis=1) # distance of source pair
        dis_t = np.linalg.norm(pcj1 - pcj2,axis=1) # distance of target pair
        d_i1i2j1j2 = np.power(dis_s - dis_t,2)     # difference between src and tgt
        
        # filter the corrspondence pair which is too close, and which doesn't reserve distance invariant
        filterIdx = np.logical_and(d_i1i2j1j2 < np.power(para.distThre,2),np.minimum(dis_s,dis_t) > 1.5*np.power(para.distSepThre,2))

        if filterIdx.sum() < 3:
            print("Stage 1: Too few satisfied pairs!")
            return np.eye(4)

        # collect index that passed the distance test
        idx = idx[filterIdx]
        idy = idy[filterIdx]
        pci1 = pci1[filterIdx]
        pcj1 = pcj1[filterIdx]
        pci2 = pci2[filterIdx]
        pcj2 = pcj2[filterIdx]
        ni1 = ni1[filterIdx]
        nj1 = nj1[filterIdx]
        ni2 = ni2[filterIdx]
        nj2 = nj2[filterIdx]
        d_i1i2j1j2 = d_i1i2j1j2[filterIdx]
    
        # angle consistency
        e1 = (pci1-pci2)
        e2 = (pcj1-pcj2)
        e1 /= np.linalg.norm(e1,axis=1,keepdims=True)
        e2 /= np.linalg.norm(e2,axis=1,keepdims=True)

        
        alpha_i1i2j1j2 = np.power(np.arccos((ni1*ni2).sum(1).clip(-1,1)) - np.arccos((nj1*nj2).sum(1).clip(-1,1)),2)
        beta_i1i2j1j2 = np.power(np.arccos((ni1*e1).sum(1).clip(-1,1)) - np.arccos((nj1*e2).sum(1).clip(-1,1)),2)
        gamma_i1i2j1j2 = np.power(np.arccos((ni2*e1).sum(1).clip(-1,1)) - np.arccos((nj2*e2).sum(1).clip(-1,1)),2)
        
        filterIdx = np.logical_and.reduce((alpha_i1i2j1j2 < np.power(para.angleThre,2),
                        beta_i1i2j1j2 < np.power(para.angleThre,2),
                        gamma_i1i2j1j2 < np.power(para.angleThre,2)))
        


        if filterIdx.sum() < 3:
            print("Stage 2: Too few satisfied pairs!")
            return np.eye(4)

        # collect index that passed the angle test

        idx = idx[filterIdx]
        idy = idy[filterIdx]
        d_i1i2j1j2 = d_i1i2j1j2[filterIdx]
        alpha_i1i2j1j2 = alpha_i1i2j1j2[filterIdx]
        beta_i1i2j1j2 = beta_i1i2j1j2[filterIdx]
        gamma_i1i2j1j2 = gamma_i1i2j1j2[filterIdx]
        
        f_i1j1 = wij[corres[0,idy],corres[1,idy]]
        f_i2j2 = wij[corres[0,idx],corres[1,idx]]

        
        # compute the weight of correspondence pair
        w_i1i2j1j2 = f_i1j1*f_i2j2*np.exp(-d_i1i2j1j2/(2*para.sigmaDist**2)\
            -alpha_i1i2j1j2/(2*para.sigmaAngle1**2)\
            -beta_i1i2j1j2/(2*para.sigmaAngle2**2)\
            -gamma_i1i2j1j2/(2*para.sigmaAngle2**2))

        pi1=sourcePC[corres[0,idy],:]
        pj1=targetPC[corres[1,idy],:]
        pi2=sourcePC[corres[0,idx],:]
        pj2=targetPC[corres[1,idx],:]
        ni1=sourceNormal[corres[0,idy],:]
        nj1=targetNormal[corres[1,idy],:]
        ni2=sourceNormal[corres[0,idx],:]
        nj2=targetNormal[corres[1,idx],:]


        allSP = np.concatenate((pi1,pi2))
        allTP = np.concatenate((pj1,pj2))
        allSN = np.concatenate((ni1,ni2))
        allTN = np.concatenate((nj1,nj2))
        allWP = np.concatenate((w_i1i2j1j2, w_i1i2j1j2))
        allWN = allWP.copy()


        # Spectral Matching
        row = corres[0,idy]*num_t+corres[1,idy]
        col = corres[0,idx]*num_t+corres[1,idx]

        allSP_dict['360'] = allSP
        allTP_dict['360'] = allTP
        allSN_dict['360'] = allSN
        allTN_dict['360'] = allTN
        allWP_dict['360'] = allWP
        allWN_dict['360'] = allWN
        w_i1i2j1j2_dict['360'] = w_i1i2j1j2
        row_dict['360'] = row
        col_dict['360'] = col
        num_s_dict['360'] = num_s
        num_t_dict['360'] = num_t

    if 'plane' in para.hybrid_method or 'corner' in para.hybrid_method:
        # sourcePC: [8,3]
        # targetPC: [8,3]
        sourcePC = dataS['plane']['pc']
        targetPC = dataT['plane']['pc']

        sourceNormal = dataS['plane']['normal']
        targetNormal = dataT['plane']['normal']

        num_corres = sourcePC.shape[0] * targetPC.shape[0]
        num_s = sourcePC.shape[0]
        num_t = targetPC.shape[0]

        corres = np.zeros([2, num_corres])
        corres[0, :] = np.arange(num_s).repeat(num_t)
        corres[1, :] = np.tile(np.arange(num_t),num_t).flatten()
        corres = corres.astype('int')

        # get the pair of correspondence index
        idx = np.tile(np.arange(num_corres),num_corres).reshape(-1, num_corres)
        idy = idx.T
        valid = (idx > idy)
        idx = idx[valid]
        idy = idy[valid]

        # distance consistency
        pci1 = sourcePC[corres[0,idy],:]
        pcj1 = targetPC[corres[1,idy],:]
        pci2 = sourcePC[corres[0,idx],:]
        pcj2 = targetPC[corres[1,idx],:]

        ni1 = sourceNormal[corres[0,idy],:]
        nj1 = targetNormal[corres[1,idy],:]
        ni2 = sourceNormal[corres[0,idx],:]
        nj2 = targetNormal[corres[1,idx],:]
   

        dis_s = np.linalg.norm(pci1 - pci2,axis=1) # distance of source pair
        dis_t = np.linalg.norm(pcj1 - pcj2,axis=1) # distance of target pair
        d_i1i2j1j2 = np.power(dis_s - dis_t,2)     # difference between src and tgt   

        # filter the corrspondence pair which is too close, and which doesn't reserve distance invariant
        filterIdx = np.logical_and(d_i1i2j1j2 < np.power(para.distThre,2),np.minimum(dis_s,dis_t) > 1.5*np.power(para.distSepThre,2))


        if filterIdx.sum() < 3:
            print("Stage 1: Too few satisfied pairs!")


        # collect index that passed the distance test
        idx = idx[filterIdx]
        idy = idy[filterIdx]
        pci1 = pci1[filterIdx]
        pcj1 = pcj1[filterIdx]
        pci2 = pci2[filterIdx]
        pcj2 = pcj2[filterIdx]
        ni1 = ni1[filterIdx]
        nj1 = nj1[filterIdx]
        ni2 = ni2[filterIdx]
        nj2 = nj2[filterIdx]
        d_i1i2j1j2 = d_i1i2j1j2[filterIdx]
        '''
        # angle consistency
        e1 = (pci1-pci2)
        e2 = (pcj1-pcj2)
        e1 /= np.linalg.norm(e1,axis=1,keepdims=True)
        e2 /= np.linalg.norm(e2,axis=1,keepdims=True)

        
        alpha_i1i2j1j2 = np.power(np.arccos((ni1*ni2).sum(1).clip(-1,1)) - np.arccos((nj1*nj2).sum(1).clip(-1,1)),2)
        beta_i1i2j1j2 = np.power(np.arccos((ni1*e1).sum(1).clip(-1,1)) - np.arccos((nj1*e2).sum(1).clip(-1,1)),2)
        gamma_i1i2j1j2 = np.power(np.arccos((ni2*e1).sum(1).clip(-1,1)) - np.arccos((nj2*e2).sum(1).clip(-1,1)),2)
        
        filterIdx = np.logical_and.reduce((alpha_i1i2j1j2 < np.power(para.angleThre,2),
                        beta_i1i2j1j2 < np.power(para.angleThre,2),
                        gamma_i1i2j1j2 < np.power(para.angleThre,2)))
       

        if filterIdx.sum() < 3:
            print("Stage 2: Too few satisfied pairs!")
            return np.eye(4)

        idx = idx[filterIdx]
        idy = idy[filterIdx]
        d_i1i2j1j2 = d_i1i2j1j2[filterIdx]
        alpha_i1i2j1j2 = alpha_i1i2j1j2[filterIdx]
        beta_i1i2j1j2 = beta_i1i2j1j2[filterIdx]
        gamma_i1i2j1j2 = gamma_i1i2j1j2[filterIdx]
        '''
        #para.sigmaAngle2 = 0.523 * 2
        # compute the weight of correspondence pair
        #w_i1i2j1j2 = np.exp(-d_i1i2j1j2/(2*para.sigmaDist**2)\
        #    -alpha_i1i2j1j2/(2*para.sigmaAngle1**2)\
        #    -beta_i1i2j1j2/(2*para.sigmaAngle2**2)\
        #    -gamma_i1i2j1j2/(2*para.sigmaAngle2**2))       
        
        #w_i1i2j1j2 = np.exp(-d_i1i2j1j2/(2*para.sigmaDist**2))
       
        # handcraft for weight 
        w_i1i2j1j2 = np.zeros(idx.shape)
        w_i1i2j1j2 += 0.01
        w_i1i2j1j2[idx==0] += 0.5
        w_i1i2j1j2[idy==0] += 0.5
        w_i1i2j1j2[idx==9] += 0.5
        w_i1i2j1j2[idy==9] += 0.5
        w_i1i2j1j2[idx==18] += 0.5
        w_i1i2j1j2[idy==18] += 0.5
        w_i1i2j1j2[idx==27] += 0.5
        w_i1i2j1j2[idy==27] += 0.5
        w_i1i2j1j2[idx==36] += 0.5
        w_i1i2j1j2[idy==36] += 0.5
        w_i1i2j1j2[idx==45] += 0.5
        w_i1i2j1j2[idy==45] += 0.5
        w_i1i2j1j2[idx==54] += 0.5
        w_i1i2j1j2[idy==54] += 0.5
        w_i1i2j1j2[idx==63] += 0.5
        w_i1i2j1j2[idy==63] += 0.5
        
        w_i1i2j1j2 *= 20 # add the weight of plane representation

        pi1=sourcePC[corres[0,idy],:]
        pj1=targetPC[corres[1,idy],:]
        pi2=sourcePC[corres[0,idx],:]
        pj2=targetPC[corres[1,idx],:]
        ni1=sourceNormal[corres[0,idy],:]
        nj1=targetNormal[corres[1,idy],:]
        ni2=sourceNormal[corres[0,idx],:]
        nj2=targetNormal[corres[1,idx],:]

        allSP = np.concatenate((pi1,pi2))
        allTP = np.concatenate((pj1,pj2))
        allSN = np.concatenate((ni1,ni2))
        allTN = np.concatenate((nj1,nj2))
        allWP = np.concatenate((w_i1i2j1j2, w_i1i2j1j2))
        allWN = allWP.copy()

        row = corres[0,idy]*num_t + corres[1,idy]
        col = corres[0,idx]*num_t + corres[1,idx]
        
        allSP_dict['plane'] = allSP
        allTP_dict['plane'] = allTP
        allSN_dict['plane'] = allSN
        allTN_dict['plane'] = allTN
        allWP_dict['plane'] = allWP
        allWN_dict['plane'] = allWN
        w_i1i2j1j2_dict['plane'] = w_i1i2j1j2
        row_dict['plane'] = row
        col_dict['plane'] = col
        num_s_dict['plane'] = num_s
        num_t_dict['plane'] = num_t

        #import pdb; pdb.set_trace()
    if spectral_method == 'original':
        return fit_spectral(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,para.mu,row_dict,col_dict,num_s_dict,num_t_dict, para.hybrid_method)
    elif spectral_method == 'original_v2':
        return fit_spectral_v2(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,para.mu,row_dict,col_dict,num_s_dict,num_t_dict, para.hybrid_method)
    elif spectral_method == 'irls_sm_v2':
        return fit_irls_sm_v2(allSP_dict,allTP_dict,allSN_dict,allTN_dict,allWP_dict,allWN_dict,w_i1i2j1j2_dict,para.mu,row_dict,col_dict,num_s_dict,num_t_dict, para.hybrid_method)
    else:
        print("No such implementation!")

def Hybrid_Spectral_Matching_M(dataS, dataT, spectral_method, R_gt_44, hybrid_method, para, fp=None):

    sourcePC_list = [] 
    targetPC_list = []

    sourceNormal_list = []
    targetNormal_list = []
    
    sourceNum_list = []
    targetNum_list = []

    candCorrs_list = [] # [3,n]
    
    corrs_num = [0]
    hybrid_representation = []
    
    points_num = [0]
    points_tgt_num = [0]

    if '360' in hybrid_method:
        hybrid_representation.append('360')
        sourcePC = dataS['360']['pc']
        targetPC = dataT['360']['pc']

        sourceNormal = dataS['360']['normal']
        targetNormal = dataT['360']['normal']

        sourceFeat = dataS['360']['feat']
        targetFeat = dataT['360']['feat']

        pano_num_s = sourcePC.shape[0] # number of source keypoints
        pano_num_t = targetPC.shape[0] # number of target keypoints
        
        sourceNum_list.append(pano_num_s)
        targetNum_list.append(pano_num_t)

        if pano_num_s < 3 or pano_num_t < 3:
            print("Too few keypoints!")
            return np.eye(4)
    
        # feature correspondence
        topK = para.topK

        t_knn = NearestNeighbors(n_neighbors=topK).fit(targetFeat)
        DIS_st, Rows_st = t_knn.kneighbors(sourceFeat)
        s_knn = NearestNeighbors(n_neighbors=topK).fit(sourceFeat)
        DIS_ts, Cols_ts = s_knn.kneighbors(targetFeat)
        #import pdb; pdb.set_trace() 
        Cols_st = np.kron(np.arange(pano_num_s),np.ones((1,topK))).reshape(pano_num_s, topK).astype('int')
        Rows_ts = np.kron(np.arange(pano_num_t),np.ones((1,topK))).reshape(pano_num_t, topK).astype('int')
    
        Rows = np.concatenate([Rows_st, Rows_ts], axis=0).T
    
        Cols = np.concatenate([Cols_st, Cols_ts], axis=0).T
        Vals = np.ones((topK, Rows.shape[1]))
    
        CorresMat = csc_matrix((Vals.flatten(), (Rows.flatten(),Cols.flatten())), shape=(pano_num_t, pano_num_s)) 
        targetIds, sourceIds = CorresMat.nonzero()
        #import pdb; pdb.set_trace()
        featureDif = sourceFeat[sourceIds, :] - targetFeat[targetIds, :]
        featureSqrDis = np.sum(np.power(featureDif, 2), axis=1)

        sigmaSqrDis = max(1e-3, np.median(featureSqrDis))
    
        featureWeights = np.exp(-featureSqrDis/(2*sigmaSqrDis))
        candCorrIds = featureWeights > 0.1
        #import pdb; pdb.set_trace()
        # get the correspondence pass the feature test
        candCorrs = np.concatenate([sourceIds[np.newaxis, candCorrIds], targetIds[np.newaxis, candCorrIds], featureWeights[np.newaxis, candCorrIds]], axis=0)
        #candCorrs = np.concatenate((candCorrs, featureSqrDis[None, candCorrIds]))
        #candCorrs = np.concatenate((candCorrs, np.array([sigmaSqrDis]*candCorrs.shape[1])[None,:]))
        candCorrs = candCorrs[:,candCorrs[0].argsort()]
        #import pdb; pdb.set_trace()

        sourcePC_list.append(sourcePC)
        targetPC_list.append(targetPC)
        sourceNormal_list.append(sourceNormal)
        targetNormal_list.append(targetNormal)

        candCorrs_list.append(candCorrs)
        corrs_num.append(candCorrs.shape[1])
        points_num.append(sourcePC.shape[0])
        points_tgt_num.append(targetPC.shape[0])

        #import pdb; pdb.set_trace()      

    if 'plane' in hybrid_method or 'corner' in hybrid_method:
        # sourcePC: [6,3] or [8,3] or [14,3]
        # targetPC: [6,3] or [8,3] or [14,3]
        hybrid_representation.append('plane')

        sourcePC_p = dataS['plane']['pc']
        targetPC_p = dataT['plane']['pc']

        sourceNormal_p = dataS['plane']['normal']
        targetNormal_p = dataT['plane']['normal']

        num_s_p = sourcePC_p.shape[0]
        num_t_p = targetPC_p.shape[0]
        
        sourceNum_list.append(num_s_p)
        targetNum_list.append(num_t_p)
        topK = para.topK
        corrs_num_p = 0

        corres_inner = np.zeros([3, num_s_p*num_t_p])
        corres_inner[0, :] = np.arange(num_s_p).repeat(num_t_p)
        corres_inner[0, :] += pano_num_s if '360' in hybrid_method else 0
        corres_inner[0, :] = corres_inner[0, :].astype('int')
        corres_inner[1, :] = np.tile(np.arange(num_t_p),num_s_p).flatten()
        corres_inner[1, :] += pano_num_t if '360' in hybrid_method else 0
        corres_inner[1, :] = corres_inner[1, :].astype('int')
        corres_inner[2, :] = para.w_plane

        #import pdb; pdb.set_trace()
        corrs_num_p += corres_inner.shape[1]
        
        # preserve for feature prune
        if 0:
            t_knn = NearestNeighbors(n_neighbors=topK).fit(targetPC_p)
            DIS_st, Rows_st = t_knn.kneighbors(sourcePC_p)
            s_knn = NearestNeighbors(n_neighbors=topK).fit(sourcePC_p)
            DIS_ts, Cols_ts = s_knn.kneighbors(targetPC_p) 

            Cols_st = np.kron(np.arange(num_s_p),np.ones((1,topK))).reshape(num_s_p, topK).astype('int')
            Rows_ts = np.kron(np.arange(num_t_p),np.ones((1,topK))).reshape(num_t_p, topK).astype('int')
        
            Rows = np.concatenate([Rows_st, Rows_ts], axis=0).T
        
            Cols = np.concatenate([Cols_st, Cols_ts], axis=0).T
            Vals = np.ones((topK, Rows.shape[1]))

            CorresMat = csc_matrix((Vals.flatten(), (Rows.flatten(),Cols.flatten())), shape=(num_t_p, num_s_p)) 
            targetIds, sourceIds = CorresMat.nonzero()

            pcDif = sourcePC_p[sourceIds, :] - targetPC_p[targetIds, :]
            pcSqrDis = np.sum(np.power(pcDif, 2), axis=1)
            sigmaSqrDis = np.median(pcSqrDis)
            featureWeights = np.exp(-pcSqrDis/(2*sigmaSqrDis))
            #import pdb; pdb.set_trace()
            candCorrIds = featureWeights > 0.2

            sourceIds = sourceIds[np.newaxis, candCorrIds]
            sourceIds += pano_num_s if '360' in hybrid_method else 0
            targetIds = targetIds[np.newaxis, candCorrIds]
            targetIds += pano_num_t if '360' in hybrid_method else 0

            candCorrs = np.concatenate([sourceIds, targetIds, featureWeights[np.newaxis, candCorrIds]], axis=0)
            
            candCorrs = candCorrs[:,candCorrs[0].argsort()]

            corres_inner = candCorrs
            corrs_num_p = corres_inner.shape[1]


        candCorrs_list.append(corres_inner)
        corrs_num.append(corrs_num_p)
        points_num.append(sourcePC_p.shape[0])
        points_tgt_num.append(targetPC_p.shape[0])

        sourcePC_list.append(sourcePC_p)
        targetPC_list.append(targetPC_p)
        sourceNormal_list.append(sourceNormal_p)
        targetNormal_list.append(targetNormal_p)   
   
    if 'topdown' in hybrid_method:
        hybrid_representation.append('topdown')
        sourcePC = dataS['topdown']['pc']
        targetPC = dataT['topdown']['pc']
        #import pdb; pdb.set_trace()
        sourceNormal = dataS['topdown']['normal']
        targetNormal = dataT['topdown']['normal']
        
        sourceNormal = np.tile(sourceNormal, (sourcePC.shape[0],1))
        targetNormal = np.tile(targetNormal, (targetPC.shape[0],1))

        sourceFeat = dataS['topdown']['feat'].T
        targetFeat = dataT['topdown']['feat'].T
 
        topdown_num_s = sourcePC.shape[0] # number of source keypoints
        topdown_num_t = targetPC.shape[0] # number of target keypoints
        
        sourceNum_list.append(topdown_num_s)
        targetNum_list.append(topdown_num_t)

        if topdown_num_s < 3 or topdown_num_t < 3:
            print("Too few keypoints!")
            return np.eye(4)

        # feature correspondence
        topK = 5
        #import pdb; pdb.set_trace()
        t_knn = NearestNeighbors(n_neighbors=topK).fit(targetFeat)
        DIS_st, Rows_st = t_knn.kneighbors(sourceFeat)
        s_knn = NearestNeighbors(n_neighbors=topK).fit(sourceFeat)
        DIS_ts, Cols_ts = s_knn.kneighbors(targetFeat)
        
        # use the groundtruth to test
        if 0:
            #pc_s2t = (np.matmul(R_gt_44[:3,:3], sourcePC.T) + R_gt_44[:3,3:4]).T
            #import pdb; pdb.set_trace()
            #R_gt_44 = dataS['topdown']['Rst']
            pc_s2t = (np.matmul(R_gt_44[:3,:3], sourcePC.T) + R_gt_44[:3,3:4]).T
            # visualzie the floor points
            if 0:
                color_s2t = np.tile(np.random.rand(3)[None,:],[pc_s2t.shape[0],1])
                color_s = np.tile(np.random.rand(3)[None,:],[sourcePC.shape[0],1]) 
                color_t = np.tile(np.random.rand(3)[None,:],[targetPC.shape[0],1])  
                util.write_ply('../visualization/s2t_s_t.ply', np.concatenate([pc_s2t,sourcePC,targetPC]),color=np.concatenate([color_s2t, color_s, color_t]))
                util.write_ply('../visualization/s2t_t.ply', np.concatenate([pc_s2t,targetPC]),color=np.concatenate([color_s2t,color_t]))
            #import pdb; pdb.set_trace()           
            
            t_knn = NearestNeighbors(n_neighbors=topK).fit(targetPC)
            #DIS_st, Rows_st = t_knn.kneighbors(sourcePC)
            DIS_st_, Rows_st_ = t_knn.kneighbors(pc_s2t)
            #s_knn = NearestNeighbors(n_neighbors=topK).fit(sourcePC)
            s_knn = NearestNeighbors(n_neighbors=topK).fit(pc_s2t)
            DIS_ts_, Cols_ts_ = s_knn.kneighbors(targetPC)

            #import pdb; pdb.set_trace()
            # test how many knn search is correct for feature
            '''
            correct_st = 0
            correct_ts = 0
            for i in range(len(Rows_st)):
                set1 = set(Rows_st[i])
                set2 = set(Rows_st_[i])
                correct_st += len(set1 & set2)

                set1 = set(Cols_ts[i])
                set2 = set(Cols_ts_[i])
                correct_ts += len(set1 & set2)
            print(correct_st / 500.)
            print(correct_ts / 500.)
            '''
            #correct_st = np.sum(np.logical_and(Rows_st, Rows_st_))
            #correct_ts = np.sum(np.logical_and(Cols_ts, Cols_ts_))

            DIS_st = DIS_st_
            DIS_ts = DIS_ts_
            Rows_st = Rows_st_
            Cols_ts = Cols_ts_


        Cols_st = np.kron(np.arange(topdown_num_s),np.ones((1,topK))).reshape(topdown_num_s, topK).astype('int')
        Rows_ts = np.kron(np.arange(topdown_num_t),np.ones((1,topK))).reshape(topdown_num_t, topK).astype('int')
    
        Rows = np.concatenate([Rows_st, Rows_ts], axis=0).T
    
        Cols = np.concatenate([Cols_st, Cols_ts], axis=0).T
        Vals = np.ones((topK, Rows.shape[1]))
    
        CorresMat = csc_matrix((Vals.flatten(), (Rows.flatten(),Cols.flatten())), shape=(topdown_num_t, topdown_num_s)) 
        targetIds, sourceIds = CorresMat.nonzero()
        
        if 1:
            featureDif = sourceFeat[sourceIds, :] - targetFeat[targetIds, :]
            featureSqrDis = np.sum(np.power(featureDif, 2), axis=1)
            sigmaSqrDis = max(1e-3, np.median(featureSqrDis))        
            featureWeights = np.exp(-featureSqrDis/(2*sigmaSqrDis))
            candCorrIds = featureWeights > 0.6
        else:
            # groundtruth test
            #pcDif = pc_s2t[sourceIds, :] - targetPC[targetIds, :]
            pcDif = sourcePC[sourceIds, :] - targetPC[targetIds, :]
            pcSqrDis = np.sum(np.power(pcDif, 2), axis=1)
            sigmaSqrDis = np.median(pcSqrDis)
            featureWeights = np.exp(-pcSqrDis/(2*sigmaSqrDis))
            candCorrIds = featureWeights > 0.2

        # get the correspondence pass the feature test

        sourceIds = sourceIds[np.newaxis, candCorrIds]
        sourceIds += pano_num_s if '360' in hybrid_method else 0
        sourceIds += num_s_p if 'plane' in hybrid_method else 0

        targetIds = targetIds[np.newaxis, candCorrIds]
        targetIds += pano_num_t if '360' in hybrid_method else 0
        targetIds += num_t_p if 'plane' in hybrid_method else 0

        featureWeights = featureWeights[np.newaxis, candCorrIds] * para.w_topdown
        candCorrs = np.concatenate([sourceIds, targetIds, featureWeights], axis=0)
        candCorrs = candCorrs[:,candCorrs[0].argsort()]
        #import pdb; pdb.set_trace()

        sourcePC_list.append(sourcePC)
        targetPC_list.append(targetPC)
        sourceNormal_list.append(sourceNormal)
        targetNormal_list.append(targetNormal)

        candCorrs_list.append(candCorrs)
        corrs_num.append(candCorrs.shape[1])     
        points_num.append(sourcePC.shape[0])
        points_tgt_num.append(targetPC.shape[0])

    if len(sourceNormal_list)==1:
        sourcePC_list = sourcePC_list[0]
        targetPC_list = targetPC_list[0]
        sourceNormal_list = sourceNormal_list[0]
        targetNormal_list = targetNormal_list[0]
        candCorrs_list = candCorrs_list[0] 
    else:

        sourcePC_list = np.concatenate(sourcePC_list, axis=0)
        targetPC_list = np.concatenate(targetPC_list, axis=0)
        sourceNormal_list = np.concatenate(sourceNormal_list, axis=0)
        targetNormal_list = np.concatenate(targetNormal_list, axis=0)
        candCorrs_list = np.concatenate(candCorrs_list, axis=1)
    
        for k in range(1, len(corrs_num)):
            corrs_num[k] += corrs_num[k-1]
        for k in range(1, len(points_num)):
            points_num[k] += points_num[k-1]
        for k in range(1, len(points_tgt_num)):
            points_tgt_num[k] += points_tgt_num[k-1]
    '''
    np.save(fp, {'sourcePC_list':sourcePC_list.T, 'targetPC_list':targetPC_list.T, 'sourceNormal_list':sourceNormal_list.T,
            'targetNormal_list':targetNormal_list.T, 'candCorrs_list':candCorrs_list, 'corrs_num':corrs_num,
            'hybrid_representation':hybrid_representation, 'R_gt':R_gt_44})
    '''

    matC = consistency_matrix(sourcePC_list.T, targetPC_list.T, sourceNormal_list.T, targetNormal_list.T, candCorrs_list, corrs_num, hybrid_representation, para)
    
    matches = leading_eigens(matC, para.numMatches)
    
    #import pdb; pdb.set_trace()
    R_hat_l = []
    overlap_val = []
    #import pdb; pdb.set_trace()
    return_corres = candCorrs_list[0:2, matches[0]]

    for k in range(para.numMatches):
        R_opt, t_opt = rigid_refinement(sourcePC_list.T, targetPC_list.T, sourceNormal_list.T, targetNormal_list.T, candCorrs_list[0:2, matches[k]], para.w_nor, para.w_fea, para.max_dist, spectral_method, sourceNum_list, targetNum_list)

        R_44_tmp = np.eye(4)
        R_44_tmp[:3,:3] = R_opt
        R_44_tmp[:3,3] = t_opt

        overlap_val_, _, _, _ = util.point_cloud_overlap(sourcePC_list, targetPC_list, R_44_tmp)
        R_hat_l.append(R_44_tmp)
        overlap_val.append(overlap_val_)
    
    if para.draw_corres == 0:
        return R_hat_l, np.asarray(overlap_val)
    else:
        return R_hat_l, np.asarray(overlap_val), sourcePC_list, targetPC_list, return_corres, points_num, points_tgt_num

def eval_Spectral():
   para = opts()
   dataS = sio.loadmat('./source.mat')
   dataT = sio.loadmat('./target.mat')
   num_data = dataS['R'].shape[0]
   ads = 0
   count = 0
   for j in range(num_data):
       R_gt = np.matmul(dataT['R'][j], np.linalg.inv(dataS['R'][j]))
       dataS_tmp = {}
       dataS_tmp['pc'] = dataS['pc'][0][j]
       dataS_tmp['normal'] = dataS['normal'][0][j]
       dataS_tmp['feat'] = dataS['feat'][0][j]
       dataT_tmp = {}
       dataT_tmp['pc'] = dataT['pc'][0][j]
       dataT_tmp['normal'] = dataT['normal'][0][j]
       dataT_tmp['feat'] = dataT['feat'][0][j]



       overlap_val,cam_dist_this,pc_dist_this,pc_nn = util.point_cloud_overlap(pc_src, pc_tgt, R_gt_44)
       overlap = '0-0.1' if overlap_val <= 0.1 else '0.1-0.5' if overlap_val <= 0.5 else '0.5-1.0'
      

       #import pdb;pdb.set_trace()
       R = Spectral_Matching(dataS_tmp, dataT_tmp, 'k_means', para)
       count += 1
       if isinstance(R, list):
           ad_min = 360
           for i in range(len(R)):
               ad_tmp = angular_distance_np(R[i][:3,:3].reshape(1,3,3), R_gt[:3,:3].reshape(1,3,3))[0]
               if ad_tmp < ad_min:
                   print("min rp module:", i, ad_tmp)
                   ad_min = ad_tmp
           ads += ad_min
           print(j, ads / count)
       else:
           #import pdb;pdb.set_trace()
           ad_tmp = angular_distance_np(R[:3,:3].reshape(1,3,3), R_gt[:3,:3].reshape(1,3,3))[0]
           print(ad_tmp)
           ads += ad_tmp
           print(j, ads / count)

   print("average angular distance:", ads/count)

if __name__ == '__main__':
    eval_Spectral()
