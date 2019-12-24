
import numpy as np 
import os 
from scipy.sparse import csr_matrix
from sklearn.neighbors import KDTree
from scipy.linalg import expm
import sys
sys.path.append('../')
import util
import scipy.io as sio
lambda_0 = 1. 
lambda_1 = 1.
lambda_2 = 1.
# \lambda_0 * ((Rp_s+t-p_t)'n_t)^2 +##-- \lambda_0 * ((Rp_s+t-p_t)'Rn_s)^2 --##
# + \lambda_1 * (1 - ((Rn_s)'n_t)^2)^2
# + \lambda_2 * ((Rn_s)'n_t)^2
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


def objective(pos_nn_s_this, pos_nn_t_this, nor_nn_s_this, nor_nn_t_this, data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, w_nn, w_coplane, w_parallel, w_perp, w_reg, w_reg_t, R, t):
  pos_s = data_s['pos']
  pos_t = data_t['pos']
  nor_s = data_s['nor']
  nor_t = data_t['nor']
  obj = 0
  obj1,obj2,obj3=0,0,0
  obj_nn=0
  stats = {}
  stats_perp=[]
  stats_nn = []
  stats_coplane = []
  stats_parallel = []
  obj_reg = w_reg * np.power(R - np.eye(3),2).sum()+w_reg_t * np.power(t,2).sum()
  if len(pos_nn_s_this):
    
    for j in range(pos_nn_s_this.shape[1]):
      p_s = pos_nn_s_this[:,j]
      n_s = nor_nn_s_this[:,j]
      p_t = pos_nn_t_this[:,j]
      n_t = nor_nn_t_this[:,j]
      # util.write_ply('test.ply', np.concatenate(((np.matmul(R, pos_nn_s_this)+t[:,None]).T, pos_nn_t_this.T)))
      stats_nn.append(np.dot(np.matmul(R, p_s) + t - p_t, n_t)**2
        + (np.dot((np.matmul(R, p_s) + t - p_t), np.matmul(R, n_s)))**2)
      tp = w_nn[j]*(stats_nn[-1])
      obj_nn += tp
  
  if len(pairID_coplane):
    
    for j in range(pairID_coplane.shape[1]):
      p_s = pos_s[:,pairID_coplane[0,j]]
      n_s = nor_s[:,pairID_coplane[0,j]]
      p_t = pos_t[:,pairID_coplane[1,j]]
      n_t = nor_t[:,pairID_coplane[1,j]]
      stats_coplane = np.dot(np.matmul(R, p_s) + t - p_t, n_t)**2 + \
        np.dot(np.matmul(R, p_s) + t - p_t, np.matmul(R, n_s))**2
      tp = w_coplane[j]*(stats_coplane)
      obj1 += tp
    
  if len(pairID_parallel):
    
    for j in range(pairID_parallel.shape[1]):
      n_s = nor_s[:,pairID_parallel[0,j]]
      n_t = nor_t[:,pairID_parallel[1,j]]
      stats_parallel.append((1 - np.dot(np.matmul(R, n_s),n_t)**2)**2)
      tp = w_parallel[j] * stats_parallel[-1]
      obj2 += tp
    
  if len(pairID_perp):
    
    for j in range(pairID_perp.shape[1]):
      n_s = nor_s[:,pairID_perp[0,j]]
      n_t = nor_t[:,pairID_perp[1,j]]
      stats_perp.append((np.dot(np.matmul(R, n_s),n_t)**2))
      tp =  w_perp[j]*stats_perp[-1]
      obj3 += tp
    
  obj=obj3+obj1+obj2+obj_nn+obj_reg
  stats['nn'] = stats_nn
  stats['perp'] = stats_perp
  stats['parallel'] = stats_parallel
  stats['coplane'] = stats_coplane
  return obj,obj1,obj2,obj3,obj_nn,obj_reg,stats
def v_cross(c):
  return np.array([[0, -c[2], c[1]],[c[2],0,-c[0]],[-c[1],c[0],0]])

def solve(data_s_nn, data_t_nn, data_s, data_t, id_s, id_t, rel_cls, R_gt, t_gt):
  mask = np.where(rel_cls == 3)[0]
  pairID_coplane = np.stack((id_s[mask], id_t[mask]), 0)
  if pairID_coplane.shape[1] == 0:
    pairID_coplane=[]
  mask = np.where(rel_cls == 2)[0]
  pairID_parallel = np.stack((id_s[mask], id_t[mask]), 0)
  if pairID_parallel.shape[1] == 0:
    pairID_parallel=[]
  mask = np.where(rel_cls == 1)[0]
  pairID_perp = np.stack((id_s[mask], id_t[mask]), 0)
  if pairID_perp.shape[1] == 0:
    pairID_perp=[]
  # transpose pos and nor
  data_s['pos'] = data_s['pos'].T
  data_s['nor'] = data_s['nor'].T
  data_t['pos'] = data_t['pos'].T
  data_t['nor'] = data_t['nor'].T
  data_s_nn_in = data_s_nn.copy()
  data_s_nn_in['pos'] = data_s_nn_in['pos'].T
  data_s_nn_in['nor'] = data_s_nn_in['nor'].T
  data_t_nn_in = data_t_nn.copy()
  data_t_nn_in['pos'] = data_t_nn_in['pos'].T
  data_t_nn_in['nor'] = data_t_nn_in['nor'].T
  
  R_pred, t_pred, e_cur, e_init, e_nn_cur, e_nn_init, stats, valid = run_optimization(data_s_nn_in, data_t_nn_in, data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, R_gt, t_gt)
  return R_pred, t_pred, e_cur, e_init, e_nn_cur, e_nn_init, stats, valid

def run_optimization(data_s_nn, data_t_nn, data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, R_gt, t_gt):
  
  pos_s = data_s['pos']
  pos_t = data_t['pos']
  nor_s = data_s['nor']
  nor_t = data_t['nor']

  pos_nn_s = data_s_nn['pos']
  pos_nn_t = data_t_nn['pos']
  nor_nn_s = data_s_nn['nor']
  nor_nn_t = data_t_nn['nor']

  tree = KDTree(pos_nn_s.T)
  nearest_dist1, nearest_ind1 = tree.query(pos_nn_t.T, k=1)
  nearest_ind1 = nearest_ind1.squeeze()
  idx_t = (nearest_dist1 < 3).squeeze()
        
  tree = KDTree(pos_nn_t.T)
  nearest_dist2, nearest_ind2 = tree.query(pos_nn_s.T, k=1)
  nearest_ind2 = nearest_ind2.squeeze()
  idx_s = (nearest_dist2 < 3).squeeze()
  
  pos_nn_s = pos_nn_s[:,idx_s]
  nor_nn_s = nor_nn_s[:,idx_s]
  pos_nn_t = pos_nn_t[:,idx_t]
  nor_nn_t = nor_nn_t[:,idx_t]

  tree = KDTree(pos_nn_t.T)

  alpha_nn = 1.0*1e-3*10  * 1e-4
  alpha_r = 1.0*0.03
  w_reg = 0.2
  w_reg_t = 0.05*30
  w_coplane = []
  w_parallel = []
  w_perp = []
  w_nn = []
  lambda_1 = 1.0 * 1e-1
  lambda_2 = 1.0
  lambda_3 = 1.0 
  if len(pairID_coplane):
    w_coplane = np.ones([pairID_coplane.shape[1]]) * lambda_1 * alpha_r
  if len(pairID_parallel):
    w_parallel = np.ones([pairID_parallel.shape[1]]) * lambda_2 * alpha_r
  if len(pairID_perp):
    w_perp = np.ones([pairID_perp.shape[1]]) * lambda_3 * alpha_r
  if len(pos_nn_t):
    nearest_dist1, nearest_ind1 = tree.query(pos_nn_s.T, k=1)
    nearest_dist1 = np.power(nearest_dist1.squeeze(), 2)
    sigma2 = sorted(nearest_dist1)[len(nearest_dist1)//4]
    w_nn = np.exp(-nearest_dist1 /2/sigma2) *  alpha_nn# / pos_nn_t.shape[1]
    w_nn = 1 / (np.sqrt(nearest_dist1+1e-4)) *  alpha_nn

  MAX_ITER_OUTER = 3
  MAX_ITER_INNER = 4
  
  if 1:
    R_cur = np.eye(3)
    t_cur = np.zeros([3])
    R_prev = R_cur
    t_prev = t_cur

    for i_outer in range(MAX_ITER_OUTER):
      dim = 6
 
      pos_nn_s_this = (np.matmul(R_cur, pos_nn_s) + t_cur[:, None])
      nearest_dist1, nearest_ind1 = tree.query(pos_nn_s_this.T, k=1)
      nearest_ind1 = nearest_ind1.squeeze()
      pos_nn_t_this = pos_nn_t[:,nearest_ind1].copy()
      nor_nn_t_this = nor_nn_t[:,nearest_ind1].copy()
    
      e_this,e1_this,e2_this,e3_this,e_nn_this,e_reg_this,stats_this = objective(pos_nn_s, pos_nn_t_this, nor_nn_s, nor_nn_t_this, data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, w_nn, w_coplane, w_parallel, w_perp, w_reg,w_reg_t, R_cur, t_cur)

      err_r = angular_distance_np(R_cur, R_gt)[0]
      err_t = np.linalg.norm(t_cur-t_gt)
      if i_outer == 0:
        e_cur = e_this
        e1_cur = e1_this
        e2_cur = e2_this
        e3_cur = e3_this
        e_reg_cur = e_reg_this
        e_nn_cur = e_nn_this
        e_init = e_this
        e_nn_init = e_nn_this
        stats_cur = stats_this

      for i_inner in range(MAX_ITER_INNER):
        
        print('outer iter %d/%d inner iter %d/%d, err_r:%.3f err_t: %.3f, current objective: %.16f, %.16f, %.16f, %.16f, %.16f, %.16f' %  (i_outer, MAX_ITER_OUTER, i_inner, MAX_ITER_INNER, err_r, err_t, e_cur, e_nn_cur, e1_cur, e2_cur, e3_cur, e_reg_cur))
        print('w_perp: %.3f, w_coplane: %.3f, w_parallel: %.3f, w_nn: %.3f' % (np.mean(w_perp), np.mean(w_coplane), np.mean(w_parallel), np.mean(w_nn)))
        # term for closest point: || \alpha_c * (((Rp_s+t-p_t)'n_t)^2 + ((Rp_s+t-p_t)'Rn_s)^2)
        # \||(R - I)\||^2 + \||t\||^2
        # \lambda_0 * ((Rp_s+t-p_t)'n_t)^2 +##-- \lambda_0 * ((Rp_s+t-p_t)'Rn_s)^2 --##
        # + \lambda_1 * (1 - ((Rn_s)'n_t)^2)^2
        # + \lambda_2 * ((Rn_s)'n_t)^2

        if len(pos_nn_s_this) > 0:

          assert(pos_nn_s_this.shape[1] == pos_nn_t_this.shape[1]==nor_nn_t_this.shape[1])
          n_nn = pos_nn_s_this.shape[1]
          J_nn = np.zeros([dim, n_nn])
          g_nn = np.zeros([n_nn])
          J_nn2 = np.zeros([dim, n_nn])
          g_nn2 = np.zeros([n_nn])
          rowsJ_nn_2 = np.tile(np.arange(n_nn),[dim, 1])
          colsJ_nn_2 = np.tile(np.arange(dim)[:,None], [1, n_nn])
          valsJ_nn_2 = np.zeros([dim, n_nn])
          g_nn_2 = np.zeros([n_nn])
          # queryCur = np.matmul(R_cur, pos_nn_s) + t_cur[:,None]
          queryCur = np.matmul(R_cur, pos_nn_s)
          J_nn[:3, :] = np.cross(queryCur.T, nor_nn_t_this.T).T 
          J_nn[3:6, :] = nor_nn_t_this
          g_nn = ((np.matmul(R_cur, pos_nn_s) + t_cur[:,None] - pos_nn_t_this) * nor_nn_t_this).sum(0)
          n_tp = np.matmul(R_cur, nor_nn_s)
          J_nn2[:3, :] = np.cross(n_tp.T, (t_cur[:,None]-pos_nn_t_this).T).T 
          J_nn2[3:6, :] = n_tp 
          g_nn2 = np.multiply(np.matmul(R_cur, pos_nn_s) + t_cur[:,None] - pos_nn_t_this, np.matmul(R_cur,nor_nn_s)).sum(0)

          if 0:
            for j in range(n_nn):
              valsJ_nn_2[:3, j] = -np.matmul(t_cur-p_t, v_cross(np.matmul(R_cur, n_s)))
              valsJ_nn_2[3:6, j] = np.matmul(R_cur, n_s)
              g_nn_2[j] = np.dot(np.matmul(R_cur, p_s) + t_cur - p_t, np.matmul(R_cur,n_s))

        n0 = 9
        rowsJ0 = np.tile(np.arange(n0),[3, 1])
        colsJ0 = np.tile(np.arange(3)[:,None], [1, n0])
        valsJ0 = np.zeros([3, n0])
        g0 = np.zeros([n0])
        valsJ0[:3, :3] = -v_cross(R_cur[:, 0])
        valsJ0[:3, 3:6] = -v_cross(R_cur[:, 1])
        valsJ0[:3, 6:9] = -v_cross(R_cur[:, 2])
        g0 = (R_cur - np.eye(3)).T.flatten()
        n0_1 = 3
        rowsJ0_1 = np.tile(np.arange(n0_1),[6, 1])
        colsJ0_1 = np.tile(np.arange(6)[:,None], [1, n0_1])
        valsJ0_1 = np.zeros([6, n0_1])
        g0_1 = np.zeros([n0_1])
        valsJ0_1[3:6, :3] = np.eye(3)
        g0_1 = t_cur.flatten()

        
        if len(pairID_coplane):
          n1 = pairID_coplane.shape[1]
          J1 = np.zeros([dim, n1])
          pos_s_trans = np.matmul(R_cur, pos_s[:,pairID_coplane[0]])
          n_s_trans = np.matmul(R_cur, nor_s[:,pairID_coplane[0]])
          J1[:3, :] = np.cross(pos_s_trans.T, nor_t[:,pairID_coplane[1]].T).T
          J1[3:6,:] = nor_t[:,pairID_coplane[1]]
          g1 = np.multiply(pos_s_trans + t_cur[:,None] - pos_t[:,pairID_coplane[1]], nor_t[:,pairID_coplane[1]]).sum(0)
          J1_2 = np.zeros([dim, n1])
          J1_2[:3, :] = np.cross(n_s_trans.T, (t_cur[:,None]-pos_t[:,pairID_coplane[1]]).T).T
          J1_2[3:6, :] = np.matmul(R_cur, nor_s[:,pairID_coplane[0]])
          g1_2 = np.multiply(pos_s_trans + t_cur[:,None] - pos_t[:,pairID_coplane[1]], n_s_trans).sum(0)
        
        if len(pairID_parallel):
          n2 = pairID_parallel.shape[1]
          J2 = np.zeros([dim, n2])
          R_ns_trans = np.matmul(R_cur, nor_s[:,pairID_parallel[0]])
          J2[:3,:] = -2*(np.sum(np.matmul(R_cur, nor_s[:,pairID_parallel[0]])* nor_t[:,pairID_parallel[1]], 0))[None,:] * np.cross(R_ns_trans.T, nor_t[:,pairID_parallel[1]].T).T
          g2 = (1 - np.multiply(np.matmul(R_cur, nor_s[:,pairID_parallel[0]]), nor_t[:,pairID_parallel[1]]).sum(0)**2)
          

        if len(pairID_perp):
          n3 = pairID_perp.shape[1]
          J3 = np.zeros([dim, n3])
          R_s_trans = np.matmul(R_cur, nor_s[:,pairID_perp[0]])
          J3[:3,:] = np.cross(R_s_trans.T, nor_t[:,pairID_perp[1]].T).T
          g3 =  np.multiply(np.matmul(R_cur, nor_s[:,pairID_perp[0]]), nor_t[:,pairID_perp[1]]).sum(0)
        #if (i_outer==0 and i_inner==0):
        #    break
        A = np.zeros([6, 6])
        b = np.zeros([6])
        J0 = np.array(csr_matrix( (valsJ0.flatten(),(rowsJ0.flatten(),colsJ0.flatten())), shape=(n0,6) ).todense())
        A += w_reg*np.matmul(J0.T, J0)
        b += -w_reg*np.matmul(J0.T, g0)
        J0_1 = np.array(csr_matrix( (valsJ0_1.flatten(),(rowsJ0_1.flatten(),colsJ0_1.flatten())), shape=(n0_1,6) ).todense())
        A +=  w_reg_t*np.matmul(J0_1.T, J0_1)
        b += -w_reg_t*np.matmul(J0_1.T, g0_1)
        print(A.max())
        if n_nn> 0:
          A += np.matmul(J_nn, np.matmul(np.diag(w_nn), J_nn.T))
          b += -np.matmul(J_nn, np.matmul(np.diag(w_nn), g_nn))
          A += np.matmul(J_nn2, np.matmul(np.diag(w_nn), J_nn2.T))
          b += -np.matmul(J_nn2, np.matmul(np.diag(w_nn), g_nn2))

        print(A.max())
        if len(pairID_coplane):
          A += np.matmul(J1, np.matmul(np.diag(w_coplane), J1.T))
          b += -np.matmul(J1, np.matmul(np.diag(w_coplane), g1))

          A += np.matmul(J1_2, np.matmul(np.diag(w_coplane), J1_2.T))
          b += -np.matmul(J1_2, np.matmul(np.diag(w_coplane), g1_2))
        print(A.max())
        if len(pairID_parallel):
          A += np.matmul(J2, np.matmul(np.diag(w_parallel), J2.T))
          b += -np.matmul(J2, np.matmul(np.diag(w_parallel), g2))
        if len(pairID_perp):
          A += np.matmul(J3, np.matmul(np.diag(w_perp), J3.T))
          b += -np.matmul(J3, np.matmul(np.diag(w_perp), g3))

        # perform line search
        
        eigVals = np.linalg.eig((A+A.T)/2)[0]
        if min(eigVals) < 1e-9*max(eigVals):
          A = A + 1e-9*max(eigVals)*np.eye(6)
        dx = np.linalg.lstsq(A, b,rcond=None)[0]

        e_cur,e1_cur,e2_cur,e3_cur,e_nn_this,e_reg_this,stats_this = objective(pos_nn_s, pos_nn_t_this, nor_nn_s, nor_nn_t_this, data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, w_nn, w_coplane, w_parallel, w_perp, w_reg, w_reg_t, R_cur, t_cur)
        alpha = 1
        flag = 0
        for step in range(10):
          c = dx[:3] * alpha 
          t = dx[3:6] * alpha 
          t_next = t_cur + t
          R_next = np.matmul(expm(v_cross(c)), R_cur)
          e_this,e1_this,e2_this,e3_this,e_nn_this,e_reg_this,stats_this = objective(pos_nn_s, pos_nn_t_this, nor_nn_s, nor_nn_t_this,data_s, data_t, pairID_coplane, pairID_parallel, pairID_perp, w_nn, w_coplane, w_parallel, w_perp, w_reg,w_reg_t, R_next, t_next)
          if e_this.astype('float32') < e_cur.astype('float32'):
            
            e_cur = e_this
            e1_cur = e1_this
            e2_cur = e2_this
            e_nn_cur = e_nn_this
            e_reg_cur = e_reg_this
            R_cur = R_next
            t_cur = t_next
            stats_cur = stats_this
            flag = 1
            break
          else:
            alpha = alpha / 2
        err_r = angular_distance_np(R_cur, R_gt)[0]
        err_t = np.linalg.norm(t_cur-t_gt)
        if flag==0:
          break

        
      error1 = np.linalg.norm(R_cur - R_prev )
      error2 = np.linalg.norm(t_cur - t_prev )
      if error1 < 1e-5 and error2 < 1e-5 and not (i_outer==0 and i_inner==0):
        break
      R_prev = R_cur
      t_prev = t_cur
      
      # update term weight

      percentile = 8
      if len(pairID_coplane):
        g1 = np.power(g1, 2)
        w_coplane = 1 / (np.sqrt(g1 + 1e-4)) *  lambda_1 * alpha_r
      if len(pairID_parallel):
        g2 =  np.power(g2, 2)
        w_parallel = 1 / (np.sqrt(g2 + 1e-4)) *  lambda_2 * alpha_r
      if len(pairID_perp):
        g3 = np.power(g3, 2)
        w_perp = 1 / (np.sqrt(g3 + 1e-4)) *  lambda_3 * alpha_r
      if len(w_nn):
        pos_nn_s_this = (np.matmul(R_cur, pos_nn_s) + t_cur[:, None])
        nearest_dist1, nearest_ind1 = tree.query(pos_nn_s_this.T, k=1)
        g_nn = nearest_dist1.squeeze()
        g_nn = np.power(g_nn, 2)
        w_nn = 1 / (np.sqrt(g_nn + 1e-4)) *  alpha_nn

  return R_cur, t_cur, e_cur, e_init, e_nn_cur, e_nn_init, stats_cur, True
