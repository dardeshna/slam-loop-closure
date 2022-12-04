
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

import cv2

# from utils import *
def wrap(x, dim):
  """ Wrap the boarder of the range image.
  """
  value = x
  if value >= dim:
    value = (value - dim)
  if value < 0:
    value = (value + dim)
  return value

def gen_normal_map(current_range, current_vertex, proj_H=64, proj_W=900):
  """ Generate a normal image given the range projection of a point cloud.
      Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
      Returns: 
        normal_data: each pixel contains the corresponding normal
  """
  normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)
  
  # iterate over all pixels in the range image
  for x in range(proj_W):
    for y in range(proj_H - 1):
      p = current_vertex[y, x][:3]
      depth = current_range[y, x]
      
      if depth > 0:
        wrap_x = wrap(x + 1, proj_W)
        u = current_vertex[y, wrap_x][:3]
        u_depth = current_range[y, wrap_x]
        if u_depth <= 0:
          continue
        
        v = current_vertex[y + 1, x][:3]
        v_depth = current_range[y + 1, x]
        if v_depth <= 0:
          continue
        
        u_norm = (u - p) / np.linalg.norm(u - p)
        v_norm = (v - p) / np.linalg.norm(v - p)
        
        w = np.cross(v_norm, u_norm)
        norm = np.linalg.norm(w)
        if norm > 0:
          normal = w / norm
          normal_data[y, x] = normal
  
  return normal_data

def project_to_image(points,proj_W = 900,proj_H = 64,fov_up=3.0, fov_down=-25.0):
    depth = np.linalg.norm(points, 2, axis=1)
    points = points[(depth > 0) & (depth < 50)]  
    depth = depth[(depth > 0) & (depth < 50)]

    fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
    fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    theta = -np.arctan2(y, x)
    phi = np.arcsin(z/depth)


    proj_x = proj_W/(2*np.pi)  * (theta + np.pi)
    proj_y = (1.0 - (phi + abs(fov_down)) / fov) *proj_H 

    proj_x = np.maximum(0, np.minimum(proj_W - 1, np.floor(proj_x))).astype(np.int32)  # in [0,W-1]
    proj_y = np.maximum(0, np.minimum(proj_H - 1, np.floor(proj_y))).astype(np.int32)  # in [0,H-1]

    # order in decreasing depth
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    proj_y = proj_y[order]
    proj_x = proj_x[order]

    x = x[order]
    y = y[order]
    z = z[order]
    
    proj_range = np.full((64, 900), -1,
                       dtype=np.float32)
    proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
    proj_range[proj_y, proj_x] = depth
    proj_vertex[proj_y, proj_x] = np.array([x, y, z, np.ones(len(x))]).T

    return proj_range,proj_vertex

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data

def project_velo_to_cam2(lidcalib,cam_calib):
    P_lid2cam_ref = np.block([[lidcalib['R'].reshape(3, 3),lidcalib['T'].reshape(3,1)],[ np.array([0., 0., 0., 1.])]])  # velo2ref_cam
    R_ref2rect = np.eye(4)
    R0_rect = cam_calib['R_rect_00'].reshape(3, 3)  # ref_cam2rect
    R_ref2rect[:3, :3] = R0_rect
    P_rect2cam2 = cam_calib['P_rect_02'].reshape((3, 4))
    proj_mat = P_rect2cam2 @ R_ref2rect @ P_lid2cam_ref
    return proj_mat

def read_calib_file(filepath):
    """
    Read in a calibration file and parse into a dictionary.
    Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
    """
    data = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            line = line.rstrip()
            if len(line) == 0: continue
            key, value = line.split(':', 1)
            # The only non-float values in these files are dates, which
            # we don't care about anyway
            try:
                data[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass
    return data


def range_projection(current_vertex, fov_up=3.0, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50):
  """ Project a pointcloud into a spherical projection, range image.
      Args:
        current_vertex: raw point clouds
      Returns: 
        proj_range: projected range image with depth, each pixel contains the corresponding depth
        proj_vertex: each pixel contains the corresponding point (x, y, z, 1)
        proj_intensity: each pixel contains the corresponding intensity
        proj_idx: each pixel contains the corresponding index of the point in the raw point cloud
  """
  # laser parameters
  fov_up = fov_up / 180.0 * np.pi  # field of view up in radians
  fov_down = fov_down / 180.0 * np.pi  # field of view down in radians
  fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians
  
  # get depth of all points
  depth = np.linalg.norm(current_vertex[:, :3], 2, axis=1)
  current_vertex = current_vertex[(depth > 0) & (depth < max_range)]  # get rid of [0, 0, 0] points
  depth = depth[(depth > 0) & (depth < max_range)]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  intensity = current_vertex[:, 3]
  
  # get angles of all points
  yaw = -np.arctan2(scan_y, scan_x)
  pitch = np.arcsin(scan_z / depth)
  
  # get projections in image coords
  proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
  proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]
  
  # scale to image size using angular resolution
  proj_x *= proj_W  # in [0.0, W]
  proj_y *= proj_H  # in [0.0, H]
  
  # round and clamp for use as index
  proj_x = np.floor(proj_x)
  proj_x = np.minimum(proj_W - 1, proj_x)
  proj_x = np.maximum(0, proj_x).astype(np.int32)  # in [0,W-1]
  
  proj_y = np.floor(proj_y)
  proj_y = np.minimum(proj_H - 1, proj_y)
  proj_y = np.maximum(0, proj_y).astype(np.int32)  # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  intensity = intensity[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  indices = np.arange(depth.shape[0])
  indices = indices[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  proj_idx = np.full((proj_H, proj_W), -1,
                     dtype=np.int32)  # [H,W] index (-1 is no data)
  proj_intensity = np.full((proj_H, proj_W), -1,
                     dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  proj_idx[proj_y, proj_x] = indices
  proj_intensity[proj_y, proj_x] = intensity
  
  return proj_range, proj_vertex, proj_intensity, proj_idx



def generate_projections(velo_path,depth_dst_folder,normal_dst_folder):

    depth_dst_folder = os.path.join(depth_dst_folder, 'depth')
    normal_dst_folder = os.path.join(normal_dst_folder, 'normal')
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


    for i in range(len(os.listdir(velo_path))):

        bin_pcd = np.fromfile(velo_path+"{i}.bin".format(i=str(i).zfill(10)), dtype = np.float32)
        depths = []
        normals = []
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        proj_range, proj_vertex = project_to_image(points)
        normal_data = gen_normal_map(proj_range, proj_vertex)
    
        # generate the destination path
        depth_dst_path = os.path.join(depth_dst_folder, str(i).zfill(6))
        normal_dst_path = os.path.join(normal_dst_folder, str(i).zfill(6))
        
        # save the semantic image as format of .npy
        np.save(depth_dst_path, proj_range)
        np.save(normal_dst_path, normal_data)
        depths.append(proj_range)
        normals.append(normal_data)
        print('finished generating depth data at: ', depth_dst_path)
        print('finished generating intensity data at: ', normal_dst_path)
    
    return depths,normals

def load_files(folder):
  """ Load all files in a folder and sort.
  """
  file_paths = [os.path.join(dp, f) for dp, dn, fn in os.walk(
    os.path.expanduser(folder)) for f in fn]
  file_paths.sort()
  return file_paths

if __name__ == '__main__':
    # Load image, calibration file, label bbox
    lidar_path = "./data/2011_09_26/2011_09_26_drive_0005_sync/velodyne_points/data/"
    depth_path = "./data/2011_09_26/2011_09_26_drive_0005_sync/"
    normal_path = "./data/2011_09_26/2011_09_26_drive_0005_sync/"
    depths = generate_projections(lidar_path,depth_path,normal_path)




