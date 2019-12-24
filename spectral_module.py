from open3d import *
import glob
import numpy as np
import cv2
import sys
from utils import torch_op
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import svds, eigs
from RPModule.rputil import *
from sklearn.neighbors import NearestNeighbors

import torch
import util
import copy
import logging
from sklearn.cluster import KMeans
from util import angular_distance_np
import scipy.io as sio


from RPModule.rputil import opts, greedy_rounding



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

def fit_spectral(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,mu,row,col,numFea_s,numFea_t):
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

 
    # compute center
    SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allSPc = allSP - SPmean
    TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allTPc = allTP - TPmean

    # compute R,t
    allS = np.concatenate((allSPc,allSN))
    allT = np.concatenate((allTPc,allTN))
    allW = np.concatenate((allWP*mu,allWN))
    R_hat = horn87_np(allS.T,allT.T,allW)
    t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
    

    R_cur = R_hat.squeeze()
    t_cur = t_hat.squeeze()
    for j in range(num_alter):
        r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
            np.power(np.matmul(R_cur,allSN.T)-allTN.T,2).sum(0))
    
        a_i1i2j1j2 = allWP*(offset - r_i1i2j1j2)
        a_i1i2j1j2[a_i1i2j1j2<0] = 0
        a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
        # construct the A matrix, compute most significant eigenvector
        A = csc_matrix((a_i1i2j1j2, (row,col)), shape=(numFea_s*numFea_t, numFea_s*numFea_t))
        A = A+A.T
        #import pdb; pdb.set_trace()       
        top_k = 1

        vals, u_ = sparse.linalg.eigs(A, k=top_k)
        u_ = u_[:, top_k-1]
        u_=u_.real
        u_ /= np.linalg.norm(u_)
        x_i1i2j1j2 = (u_[row]*u_[col]).squeeze()
        
        x_i1i2j1j2[x_i1i2j1j2<0] = 0
        x_i1i2j1j2 *= w_i1i2j1j2
        
        # initialize new weight 
        allW = np.tile(x_i1i2j1j2,4)
        # apply mu on weight corresponding to position
        allW[:len(allW)//2]*=mu

        # compute center
        allWP=allW[:len(allW)//2]
        SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allSPc = allSP - SPmean
        TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
        allTPc = allTP - TPmean

        allS = np.concatenate((allSPc,allSN))
        allT = np.concatenate((allTPc,allTN))
        #import pdb; pdb.set_trace()
        # compute R,t using current weight
        R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
        t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
    
        R_cur = R_tp
        t_cur = t_tp

    R        = np.eye(4)
    R[:3,:3] = R_cur
    R[:3,3]  = t_cur
    return R

def fit_spectral_v2(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,mu,row,col,numFea_s,numFea_t):
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

 
    # compute center
    SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allSPc = allSP - SPmean
    TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
    allTPc = allTP - TPmean

    # compute R,t
    allS = np.concatenate((allSPc,allSN))
    allT = np.concatenate((allTPc,allTN))
    allW = np.concatenate((allWP*mu,allWN))
    R_hat = horn87_np(allS.T,allT.T,allW)
    t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

    top_k = 15
    R_hat_l = []
    for k_ in range(top_k):
        R_cur = R_hat.squeeze()
        t_cur = t_hat.squeeze()

        for j in range(num_alter):
            r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
                np.power(np.matmul(R_cur,allSN.T)-allTN.T,2).sum(0))
    
            a_i1i2j1j2 = allWP*(offset - r_i1i2j1j2)
            a_i1i2j1j2[a_i1i2j1j2<0] = 0
            a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
            # construct the A matrix, compute most significant eigenvector
            A = csc_matrix((a_i1i2j1j2, (row,col)), shape=(numFea_s*numFea_t, numFea_s*numFea_t))
            A = A+A.T     

            vals, u_ = sparse.linalg.eigs(A, k=top_k)
            u_ = u_[:, k_]
            u_=u_.real
            u_ /= np.linalg.norm(u_)
            x_i1i2j1j2 = (u_[row]*u_[col]).squeeze()
        
            x_i1i2j1j2[x_i1i2j1j2<0] = 0
            x_i1i2j1j2 *= w_i1i2j1j2
        
            # initialize new weight 
            allW = np.tile(x_i1i2j1j2,4)
            # apply mu on weight corresponding to position
            allW[:len(allW)//2]*=mu

            # compute center
            allWP=allW[:len(allW)//2]
            SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allSPc = allSP - SPmean
            TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allTPc = allTP - TPmean

            allS = np.concatenate((allSPc,allSN))
            allT = np.concatenate((allTPc,allTN))

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

def fit_irls_sm_v2(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,mu,row,col,numFea_s,numFea_t):
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
    num_reweighted = 5
    num_alter      = 1
    resSigma       = 1
    offset         = 50
    EPS            = 1e-12
    allW = np.concatenate((allWP*mu,allWN))
    
    top_k = 6
    R_hat_l = []

    for k_ in range(top_k):
        # initialize R,t
        for j in range(num_reweighted):
            # get the weight for position
            allWP=allW[:len(allW)//2]
            # compute center
            SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allSPc = allSP - SPmean
            TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
            allTPc = allTP - TPmean

            allS = np.concatenate((allSPc,allSN))
            allT = np.concatenate((allTPc,allTN))

            # compute R,t using current weight
            R_hat = horn87_np(allS.T,allT.T,allW)
            t_hat = -np.matmul(R_hat.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()
            # compute new weight
            residualPc = mu*np.power(np.matmul(R_hat.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
            residualN = np.power(np.matmul(R_hat.squeeze(),allSN.T)-allTN.T,2).sum(0)
            residual = np.concatenate((residualPc,residualN))
            allW = allW * resSigma**2/(resSigma**2+residual)

        R_cur = R_hat.squeeze()
        t_cur = t_hat.squeeze()

        # alternate between spectral method and irls
        for j in range(num_alter):
            r_i1i2j1j2 = (mu*np.power(np.matmul(R_cur,allSPc.T)-allTPc.T,2).sum(0)+ \
                np.power(np.matmul(R_cur,allSN.T)-allTN.T,2).sum(0))
        
            a_i1i2j1j2 = np.tile(w_i1i2j1j2,2)*(offset - r_i1i2j1j2)
            a_i1i2j1j2[a_i1i2j1j2<0] = 0
            a_i1i2j1j2 = a_i1i2j1j2.reshape(2,-1).sum(0)
        
            # construct the A matrix, compute most significant eigenvector
            A = csc_matrix((a_i1i2j1j2, (row,col)), shape=(numFea_s*numFea_t, numFea_s*numFea_t))
            A = A+A.T
   
            vals, u_ = sparse.linalg.eigs(A, k=top_k)
            u_ = u_[:,k_]
            u_=u_.real
            u_ /= np.linalg.norm(u_)

            x_i1i2j1j2 = (u_[row]*u_[col]).squeeze()

            x_i1i2j1j2[x_i1i2j1j2<0] = 0
            x_i1i2j1j2 *= w_i1i2j1j2

            # get new weight
            allW = np.tile(x_i1i2j1j2,4)
            # apply mu on weight corresponding to position
            allW[:len(allW)//2]*=mu
        
            for l in range(num_reweighted):
                # get the weight for position
                allWP=allW[:len(allW)//2]
                # compute center
                SPmean = (allSP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
                allSPc = allSP - SPmean
                TPmean = (allTP*allWP[:,np.newaxis]).sum(0)/(allWP.sum()+EPS)
                allTPc = allTP - TPmean

                allS = np.concatenate((allSPc,allSN))
                allT = np.concatenate((allTPc,allTN))

                # compute R,t using current weight
                R_tp = horn87_np(allS.T,allT.T,allW).reshape(3,3)
                t_tp = -np.matmul(R_tp.reshape(3,3),SPmean.squeeze())+TPmean.squeeze()

                # compute new weight
                residualPc = mu*np.power(np.matmul(R_tp.squeeze(),allSPc.T)-allTPc.T,2).sum(0)
                residualN = np.power(np.matmul(R_tp.squeeze(),allSN.T)-allTN.T,2).sum(0)
                residual = np.concatenate((residualPc,residualN))
                allW = allW * resSigma**2/(resSigma**2+residual)
            R_cur = R_tp
            t_cur = t_tp


        R        = np.eye(4)
        R[:3,:3] = R_cur
        R[:3,3]  = t_cur
        R_hat_l.append(R)

    return R_hat_l

def Spectral_Matching(dataS, dataT, spectral_method, para):
    sourcePC = dataS['pc']
    targetPC = dataT['pc']

    sourceNormal = dataS['normal']
    targetNormal = dataT['normal']

    sourceFeat = dataS['feat'] / 100.
    targetFeat = dataT['feat'] / 100.
    
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

    if spectral_method == 'original':
        return fit_spectral(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,num_s,num_t)
    elif spectral_method == 'k_means':
        return fit_kmeans_sm(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,num_s,num_t,sourcePC,targetPC)

    elif spectral_method == 'update':
        return fit_new_irls_sm(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,num_s,num_t, sourcePC, targetPC)
    elif spectral_method == 'original_v2':
        return fit_spectral_v2(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,num_s,num_t)
    elif spectral_method == 'irls_sm_v2':
        return fit_irls_sm_v2(allSP,allTP,allSN,allTN,allWP,allWN,w_i1i2j1j2,para.mu,row,col,num_s,num_t)


def Spectral_Matching_M(dataS, dataT, spectral_method, para):
    sourcePC = dataS['pc']
    targetPC = dataT['pc']

    sourceNormal = dataS['normal']
    targetNormal = dataT['normal']

    sourceFeat = dataS['feat']
    targetFeat = dataT['feat']
    
    num_s = sourcePC.shape[0] # number of source keypoints
    num_t = targetPC.shape[0] # number of target keypoints

    if num_s < 3 or num_t < 3:
        print("Too few keypoints!")
        return np.eye(4)
    
    # feature correspondence
    topK = para.topK
    t_knn = NearestNeighbors(n_neighbors=topK).fit(targetFeat)
    DIS_st, Rows_st = t_knn.kneighbors(sourceFeat)
    s_knn = NearestNeighbors(n_neighbors=topK).fit(sourceFeat)
    DIS_ts, Cols_ts = s_knn.kneighbors(targetFeat)
    
    Cols_st = np.kron(np.arange(num_s),np.ones((1,topK))).reshape(num_s, topK).astype('int')
    Rows_ts = np.kron(np.arange(num_t),np.ones((1,topK))).reshape(num_t, topK).astype('int')
    
    Rows = np.concatenate([Rows_st, Rows_ts], axis=0).T
    
    Cols = np.concatenate([Cols_st, Cols_ts], axis=0).T
    Vals = np.ones((topK, Rows.shape[1]))
    
    CorresMat = csc_matrix((Vals.flatten(), (Rows.flatten(),Cols.flatten())), shape=(num_t, num_s)) 
    targetIds, sourceIds = CorresMat.nonzero()

    featureDif = sourceFeat[sourceIds, :] - targetFeat[targetIds, :]
    featureSqrDis = np.sum(np.power(featureDif, 2), axis=1)

    sigmaSqrDis = max(1e-3, np.median(featureSqrDis))
    
    featureWeights = np.exp(-featureSqrDis/(2*sigmaSqrDis))
    candCorrIds = featureWeights > 0.1
    candCorrs = np.concatenate([sourceIds[np.newaxis, candCorrIds], targetIds[np.newaxis, candCorrIds], featureWeights[np.newaxis, candCorrIds]], axis=0)
    candCorrs = candCorrs[:,candCorrs[0].argsort()]


    matC = consistency_matrix(sourcePC.T, targetPC.T, sourceNormal.T, targetNormal.T, candCorrs, para)
    
    matches = leading_eigens(matC, para.numMatches)
    R_hat_l = []

    for k in range(para.numMatches):
        R_opt, t_opt = rigid_refinement(sourcePC.T, targetPC.T, sourceNormal.T, targetNormal.T, candCorrs[0:2, matches[k]], para.w_nor, para.w_fea, para.max_dist)

        R_44_tmp = np.eye(4)
        R_44_tmp[:3,:3] = R_opt
        R_44_tmp[:3,3] = t_opt
        R_hat_l.append(R_44_tmp)

    return R_hat_l


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
