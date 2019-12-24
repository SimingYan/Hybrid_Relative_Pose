import glob
import json
import os
import cv2
import shutil
import uuid
import argparse
import random
from sklearn.neighbors import KDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import scipy 
squareform = scipy.spatial.distance.squareform
pdist = scipy.spatial.distance.pdist
import numpy as np
from sklearn.neighbors import KDTree
if os.path.exists('/scratch'):
  eldar = True
else:
  eldar = False

if eldar:
  import open3d as o3d
  read_point_cloud = o3d.io.read_point_cloud
  write_point_cloud = o3d.io.write_point_cloud
  PointCloud = o3d.geometry.PointCloud
  Vector3dVector = o3d.utility.Vector3dVector
else:
  from open3d import * 
  
scannet_color_palette= [
       (0, 0, 0),
       (174, 199, 232),		# wall
       (152, 223, 138),		# floor
       (31, 119, 180), 		# cabinet
       (255, 187, 120),		# bed
       (188, 189, 34), 		# chair
       (140, 86, 75),  		# sofa
       (255, 152, 150),		# table
       (214, 39, 40),  		# door
       (197, 176, 213),		# window
       (148, 103, 189),		# bookshelf
       (196, 156, 148),		# picture
       (23, 190, 207), 		# counter
       (178, 76, 76),  
       (247, 182, 210),		# desk
       (66, 188, 102), 
       (219, 219, 141),		# curtain
       (140, 57, 197), 
       (202, 185, 52), 
       (51, 176, 203), 
       (200, 54, 131), 
       (92, 193, 61),  
       (78, 71, 183),  
       (172, 114, 82), 
       (255, 127, 14), 		# refrigerator
       (91, 163, 138), 
       (153, 98, 156), 
       (140, 153, 101),
       (158, 218, 229),		# shower curtain
       (100, 125, 154),
       (178, 127, 135),
       (120, 185, 128),
       (146, 111, 194),
       (44, 160, 44),  		# toilet
       (112, 128, 144),		# sink
       (96, 207, 209), 
       (227, 119, 194),		# bathtub
       (213, 92, 176), 
       (94, 106, 211), 
       (82, 84, 163),  		# otherfurn
       (100, 85, 144)
    ]
scannet_color_palette = np.stack(scannet_color_palette)

def write_ply(fn, point, normal=None, color=None):
  ply = PointCloud()
  ply.points = Vector3dVector(point)
  if color is not None:
    ply.colors = Vector3dVector(color)
  if normal is not None:
    ply.normals = Vector3dVector(normal)
  write_point_cloud(fn, ply)

def plot_plane(pointcloud, plane_idx, plane_params):
  v = [pointcloud[:,:3][plane_idx ==0]]
  # v_c = [pointcloud[:,6:9][plane_idx == 0]]
  v_c = [np.tile(np.array([0,0,0])[None,:], [len(v[-1]),1])]
  for i in range(len(plane_params)):
    center = pointcloud[:,:3][plane_idx == i+1].mean(0)
    
    if (i+1) < scannet_color_palette.shape[0]:
      color = scannet_color_palette[i+1]/255.0
    else:
      color = np.random.rand(3)
    v.append(pointcloud[:,:3][plane_idx == i+1])
    v_c.append(np.tile(color[None,:], [len(v[-1]),1]))

  v = np.concatenate(v)
  v_c = np.concatenate(v_c)
  write_ply('test.ply', v, color=v_c)

def augment(xyzs):
      axyz = np.ones((len(xyzs), 4))
      axyz[:, :3] = xyzs
      return axyz
def estimate(xyzs):
  # axyz = augment(xyzs[:3])
  axyz = augment(xyzs)
  m= np.linalg.svd(axyz)[-1][-1, :]
  m /= np.linalg.norm(m[:3])
  return m
def is_inlier(coeffs, xyz, threshold):
    return np.abs(coeffs.dot(augment([xyz]).T)) < threshold
def run_ransac(tree, graph, data, indicator, estimate, is_inlier, sample_size, goal_inliers, max_iterations, principle_direction,stop_at_goal=True, random_seed=None):
  
  best_ic = 0
  best_model = None
  random.seed(random_seed)
  # random.sample cannot deal with "data" being a numpy array
  data = list(data)
  best_inlier_index = []
  data_arr = np.array(data)
  data_aug = augment(data)
  valid_id = np.where(indicator)[0]
  for i in range(max_iterations):
      # print('ransac iter: ', i)
      inlier_index = []
      # random select one point 
      seedID = valid_id[np.random.choice(len(valid_id), 1)[0]]
      seed = data[seedID]
      all_nn_indices = tree.query_radius([seed], r=0.2)[0]
      neighbors_ind = all_nn_indices[np.where(indicator[all_nn_indices])]
      if len(neighbors_ind)< int(sample_size): 
        print('continue because neighbor size too small')
        continue 
      # neighbors_ind = neighbors_ind[np.random.choice(len(neighbors_ind), int(sample_size))]
      s = data_arr[neighbors_ind]
      m = estimate(s)
      #import ipdb;ipdb.set_trace()
      #write_ply('test.ply', data_arr[neighbors_ind]+1e-3, color=np.tile(np.array([1,0,0])[None,:],[data_arr[neighbors_ind].shape[0],1]))
      
      if principle_direction is not None:
        theta = np.arccos((m[:3][None,:] * principle_direction).sum(1).clip(-1,1))/np.pi*180
        theta = np.minimum(theta, 180-theta)
        if np.min(theta) > 5:
          continue
      threshold = 0.03
      mask = np.abs((m[None, :] * data_aug).sum(1)) < threshold
      mask = mask & indicator
      # print('mask sum', mask.sum())
      # filter out disconnected components 
      ind = np.where(mask)[0]
      if len(ind) < 50: continue
      # 
      
      # 
      # dst = np.linalg.norm(all_point[:, None, :] - all_point[None, :, :], axis=2)
      # graph = (dst < 0.05).astype('int')
      
      idx_tp = np.concatenate(([seedID], ind))
      #all_point = data_arr[idx_tp]
      #dst = squareform(pdist(all_point))
      #sub_grah = (dst < 0.2).astype('int')
      
      sub_grah = graph[np.ix_(idx_tp,idx_tp)]
      sub_grah = csr_matrix(sub_grah)
      n_components, labels = connected_components(csgraph=sub_grah, directed=False, return_labels=True)
      
      ic = (labels[1:] == labels[0]).sum()
      # print('n_components', n_components, 'inlier', ic)
      
      #write_ply('test.ply',all_point)
      inlier_index = ind[np.where(labels[1:] == labels[0])[0]]

      inlier_index = inlier_index.tolist()
      
      
      #print(s)
      #print('estimate:', m,)
      #print('# inliers:', ic)

      if ic > best_ic:
          print('curre ic, ', ic)
          center =data_arr[np.array(inlier_index)].mean(0)
          #radius = np.median(np.linalg.norm(data_arr - center,axis=1))
          #score = radius**2 / np.var(data_arr[np.array(inlier_index)])
          score = np.var(data_arr[np.array(inlier_index)])

          s = [data[x] for x in inlier_index]
          
          m = estimate(s)
          # print('iter %d, goal inlier %d' % (i, goal_inliers))
          best_ic = ic
          best_model = m
          best_inlier_index = inlier_index
          print('goal inliiers:, ', goal_inliers, ic)
          if ic > goal_inliers and stop_at_goal:
              print('stop because reach goal inliers')
              break
  #print('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
  
  if i == max_iterations-1:
    print('stop because reach max iteration')
  print(score, best_ic)
  return best_model, best_ic, best_inlier_index


# do merging on plane params
    
def CheckMergeable(plane1, plane2):
  # check if parallel 
  plane1_param = plane1['param']
  plane2_param = plane2['param']
  
  dst1to2 = np.mean(np.abs((plane1['pc'] * plane2['param'][:3][None,:]).sum(1) + plane2['param'][3]))
  dst2to1 = np.mean(np.abs((plane2['pc'] * plane1['param'][:3][None,:]).sum(1) + plane1['param'][3]))
  if dst1to2 < 0.10 and dst2to1 < 0.10:
    #print('mergeable!')
    pc = np.concatenate((plane1['pc'], plane2['pc']))
    m = estimate(pc)
    new_plane={'param':np.concatenate((m,[pc.shape[0]])),'pc':pc,'plane_idx':np.concatenate((plane1['plane_idx'], plane2['plane_idx']))}
    return True, new_plane
  else:
    return False, None

def fit_planes(pc,prior_graph=None,principle_direction=None):
    plane_params = []
    plane_idx = np.zeros([pc.shape[0]])
    count = 1
    tree = KDTree(pc)
    graph = np.zeros([pc.shape[0], pc.shape[0]],dtype=bool)
    if 1:
      for i in range(pc.shape[0]):
        if i%100 == 0:print(i)
        all_nn_indices = tree.query_radius(pc[i:(i+1)], r=0.20)[0]
        graph[i, all_nn_indices] = True
      
      # graph = (graph | graph.T).astype('int')
      # graph = graph.astype('int')
      # dst = squareform(pdist(pc))
      # graph = (dst < 0.05).astype('int')
    
    if prior_graph is not None:
      graph = graph & prior_graph
    indicator = np.ones([len(pc)]).astype('bool')
    
    try:
      while True:
        #if count > 10:
        #  break
        n = indicator.sum()
        print('remaining point:', n)
        if n < 10:
          break
        max_iterations = 30
        goal_inliers = n*0.1
        print('iter')
              
        m, best_ic, best_inlier_index = run_ransac(tree, graph, pc, indicator, estimate, is_inlier, 3, goal_inliers, max_iterations,principle_direction,stop_at_goal=False)
        # m, best_ic, best_inlier_index = run_ransac(tree, graph, pc, indicator, estimate, is_inlier, 3, goal_inliers, max_iterations,principle_direction,stop_at_goal=False)
        if best_ic > 100:
          plane_params.append(np.concatenate((m,[best_ic])))
          plane_idx[best_inlier_index] = count
          count += 1 
          indicator[best_inlier_index] = False
        else:
          break
    except:
      import ipdb;ipdb.set_trace()
    continue_merging = True
    planes = []
    
    for i in range(len(plane_params)):
      planes.append({'param':plane_params[i],'pc':pc[plane_idx == i+1],'plane_idx':np.where(plane_idx == i+1)[0]})

    print("Number of planes to merge:",len(planes)) 
    while continue_merging:
      for i in range(len(planes)):
        for j in range(i+1, len(planes)):
          mergeable, new_plane = CheckMergeable(planes[i], planes[j])
          if mergeable:
            continue_merging = True
            planes = [planes[x] for x in range(len(planes)) if x not in [i,j]]
            planes.append(new_plane)
            break
          else:
            #print('not mergeable', i, j)
            continue_merging=False
        if continue_merging:break 
      

      #print('current number plane: ', len(planes))
    
    planes = sorted(planes, key=lambda x: x['pc'].shape[0])
    plane_idx = np.zeros([len(pc)])
    plane_params=[]
    for i in range(len(planes)):
      mask = np.abs((pc * planes[i]['param'][:3]).sum(1) + planes[i]['param'][3]) < 0.04
      plane_idx[planes[i]['plane_idx']] = i+1
      # plane_idx[mask] = i+1 
      plane_params.append(planes[i]['param'])
    
    return plane_params, plane_idx
