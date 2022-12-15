
import os
import numpy as np

from matplotlib import pyplot as plt

def gen_normal_map_vectorized(current_range, current_vertex, proj_H=64, proj_W=900):
  """ Generate a normal image given the range projection of a point cloud.
      Args:
        current_range:  range projection of a point cloud, each pixel contains the corresponding depth
        current_vertex: range projection of a point cloud,
                        each pixel contains the corresponding point (x, y, z, 1)
      Returns: 
        normal_data: each pixel contains the corresponding normal
  """
  normal_data = np.full((proj_H, proj_W, 3), -1, dtype=np.float32)

  current_vertex = current_vertex[...,:3]

  u = np.roll(current_vertex, -1, axis=1)[:-1]
  v = current_vertex[1:]

  current_vertex = current_vertex[:-1]

  u_norm = u - current_vertex
  v_norm = v - current_vertex

  u_norm /= np.linalg.norm(u_norm, axis=-1, keepdims=True)
  v_norm /= np.linalg.norm(v_norm, axis=-1, keepdims=True)

  w_norm = np.cross(v_norm, u_norm)
  norm = np.linalg.norm(w_norm, axis=-1, keepdims=True)

  w_norm /= norm

  mask = (current_range[:-1] > 0) & (np.roll(current_range, -1, axis=1) >= 0)[:-1] & (current_range[1:] >= 0) & (norm > 0).squeeze()

  normal_data[:-1][mask] = w_norm[mask]
  
  return normal_data

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
  depth = np.linalg.norm(current_vertex, 2, axis=1)
  mask = (depth > 0) & (depth < max_range)
  current_vertex = current_vertex[mask]  # get rid of [0, 0, 0] points
  depth = depth[mask]
  
  # get scan components
  scan_x = current_vertex[:, 0]
  scan_y = current_vertex[:, 1]
  scan_z = current_vertex[:, 2]
  
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
  proj_x = np.clip(proj_x.astype(np.int32), 0, proj_W - 1) # in [0,W-1]
  proj_y = np.clip(proj_y.astype(np.int32), 0, proj_H - 1) # in [0,H-1]
  
  # order in decreasing depth
  order = np.argsort(depth)[::-1]
  depth = depth[order]
  proj_y = proj_y[order]
  proj_x = proj_x[order]

  scan_x = scan_x[order]
  scan_y = scan_y[order]
  scan_z = scan_z[order]
  
  proj_range = np.full((proj_H, proj_W), -1,
                       dtype=np.float32)  # [H,W] range (-1 is no data)
  proj_vertex = np.full((proj_H, proj_W, 4), -1,
                        dtype=np.float32)  # [H,W] index (-1 is no data)
  
  proj_range[proj_y, proj_x] = depth
  proj_vertex[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z, np.ones(len(scan_x))]).T
  
  return proj_range, proj_vertex

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

        bin_pcd = np.fromfile(os.path.join(velo_path, "{i}.bin".format(i=str(i).zfill(6))), dtype = np.float32)
        depths = []
        normals = []
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        proj_range, proj_vertex = range_projection(points)
        normal_data = gen_normal_map_vectorized(proj_range, proj_vertex)

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


def show_images(depth_data, normal_data):
  """ This function is used to visualize different types of data
      generated from the LiDAR scan, including depth, normal, intensity and semantics.
  """
  fig, axs = plt.subplots(2, figsize=(6, 4))
  axs[0].set_title('range_data')
  axs[0].imshow(depth_data.astype(np.int32))
  axs[0].set_axis_off()
  
  axs[1].set_title('normal_data')
  axs[1].imshow(normal_data)
  axs[1].set_axis_off()

  plt.suptitle('Preprocessed data from the LiDAR scan')
  plt.show()

if __name__ == '__main__':

    sequences = [f"{i:02d}" for i in range(11)]
    # sequences = ["00"]
    base_paths = [os.path.join('data/sequences/', seq) for seq in sequences]
    lidar_paths = [os.path.join(b, 'velodyne') for b in base_paths]

    for (l, b) in zip(lidar_paths, base_paths):
      generate_projections(l, b, b)

