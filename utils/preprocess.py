
import os
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as cm
from urllib3 import encode_multipart_formdata
from yaml import scan
from com_overlap_yaw import com_overlap_yaw
from normalize_data import normalize_data
from split_train_val import split_train_val
from utils import *

def generate_embeddings(lidar_path,feature_path,sequences):
    Alldepths = []
    Allnormals = []
    for seq in sequences:
      seq_folder = os.path.join(feature_path, seq)

      try:
        os.stat(seq_folder)
      except:
        os.mkdir(seq_folder)

      depth_dst_folder = os.path.join(seq_folder,'depth')
      normal_dst_folder = os.path.join(seq_folder, 'normal')

      try:
        os.stat(depth_dst_folder)
        print('generating depth data in: ', depth_dst_folder)
      except:
        print('creating new depth folder: ', depth_dst_folder)
        os.mkdir(depth_dst_folder)

      try:
        os.stat(normal_dst_folder)
        print('generating normal data in: ', normal_dst_folder)
      except:
        print('creating new normal folder: ', normal_dst_folder)
        os.mkdir(normal_dst_folder)
      
      depths = []
      normals = []
      scan_path = os.path.join(lidar_path,seq+"/velodyne/")
      idx = 0
      for file in sorted(os.listdir(scan_path)):
        if file == ".DS_Store":
          continue
        bin_pcd = np.fromfile(os.path.join(scan_path,file), dtype = np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        depth_data, proj_vertex = generate_depth(points)
        normal_data = generate_normal(depth_data, proj_vertex)

        # generate the destination path
        depth_dst_path = os.path.join(depth_dst_folder, str(idx).zfill(6))
        normal_dst_path = os.path.join(normal_dst_folder, str(idx).zfill(6))
        
        # save the semantic image as format of .npy
        np.save(depth_dst_path, depth_data)
        np.save(normal_dst_path, normal_data)
        depths.append(depth_data)
        normals.append(normal_data)
        print('finished generating depth data at: ', depth_dst_path)
        print('finished generating intensity data at: ', normal_dst_path)
        idx+=1

    return Alldepths,Allnormals


def generate_truth(pose_path,scan_paths,feature_path,calib_path,sequences):

  for seq in sequences:
    feature_seq_path = os.path.join(feature_path,seq)
    # Load Calibration Transformation
    T_cam_velo = load_calib(os.path.join(calib_path,seq+"/calib.txt"))
    T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
    T_velo_cam = np.linalg.inv(T_cam_velo)

    #Load Poses
    poses = load_poses(os.path.join(pose_path,seq+".txt"))
    pose0_inv = np.linalg.inv(poses[0])

    poses_new = []
    for pose in poses:
      poses_new.append(T_velo_cam.dot(pose0_inv).dot(pose).dot(T_cam_velo))
      poses = np.array(poses_new)

    
    # generate overlap and yaw ground truth array
    scan_seq_path = os.path.join(scan_paths,seq+"/velodyne")
    scan_seq_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(scan_seq_path)) for f in fn]
    scan_seq_paths.sort()
    ground_truth_mapping = com_overlap_yaw(scan_seq_paths, poses, frame_idx=0)
  
    # normalize the distribution of ground truth data
    dist_norm_data = normalize_data(ground_truth_mapping)
    
    # split ground truth for training and validation
    train_data, validation_data = split_train_val(dist_norm_data)
    
    # add sequence label to the data and save them as npz files

    # specify the goal folder
    feature_seq_path = os.path.join(feature_seq_path, 'ground_truth')
    try:
      os.stat(feature_seq_path)
      print('generating depth data in: ', feature_seq_path)
    except:
      print('creating new depth folder: ', feature_seq_path)
      os.mkdir(feature_seq_path)
    
    # training data
    train_seq = np.empty((train_data.shape[0], 2), dtype=object)
    train_seq[:] = seq
    np.savez_compressed(feature_seq_path + '/train_set', overlaps=train_data, seq=train_seq)
    
    # validation data
    validation_seq = np.empty((validation_data.shape[0], 2), dtype=object)
    validation_seq[:] = seq
    np.savez_compressed(feature_seq_path + '/validation_set', overlaps=validation_data, seq=validation_seq)
    
    # raw ground truth data, fully mapping, could be used for testing
    ground_truth_seq = np.empty((ground_truth_mapping.shape[0], 2), dtype=object)
    ground_truth_seq[:] = seq
    np.savez_compressed(feature_seq_path + '/ground_truth_overlap_yaw', overlaps=ground_truth_mapping, seq=ground_truth_seq)

    vis_gt(poses[:, :2, 3], ground_truth_mapping)

def vis_gt(xys, ground_truth_mapping):
  """Visualize the overlap value on trajectory"""
  # set up plot
  fig, ax = plt.subplots()
  norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=True)
  mapper = cm.ScalarMappable(norm=norm)  # cmap="magma"
  mapper.set_array(ground_truth_mapping[:, 2])
  colors = np.array([mapper.to_rgba(a) for a in ground_truth_mapping[:, 2]])
  
  # sort according to overlap
  indices = np.argsort(ground_truth_mapping[:, 2])
  xys = xys[indices]
  
  ax.scatter(xys[:, 0], xys[:, 1], c=colors[indices], s=10)
  
  ax.axis('square')
  ax.set_xlabel('X [m]')
  ax.set_ylabel('Y [m]')
  ax.set_title('Demo 4: Generate ground truth for training')
  cbar = fig.colorbar(mapper, ax=ax)
  cbar.set_label('Overlap', rotation=270, weight='bold')
  plt.show()

if __name__ == '__main__':
    # Load image, calibration file, label bbox
    lidar_path = "./data/dataset/sequences"
    feature_path = "./data/dataset/"
    sequences = ["00"]
    pose_path = "./data/dataset/poses/"
    calib_path = "./data/dataset/calib/sequences"
    # poses = load_poses("./data/poses/00.txt")
    # depths,normals = generate_embeddings(lidar_path,feature_path,sequences)
    generate_truth(pose_path,lidar_path,feature_path,calib_path,sequences)





