
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from urllib3 import encode_multipart_formdata
from utils import *

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




def generate_embeddings(lidar_path,depth_dst_folder,normal_dst_folder):

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

    # lidarpaths = load_files(lidar_path)
  
    # for i in range(len(os.listdir(lidar_path))):
    Alldepths = []
    Allnormals = []
    for dir in sorted(os.listdir(lidar_path)):
      if dir == ".DS_Store":
        continue
      try:
        os.stat(os.path.join(depth_dst_folder,dir))
        print('generating depth data in: ', os.path.join(depth_dst_folder,dir))
      except:
        print('creating new depth folder: ', os.path.join(depth_dst_folder,dir))
        os.mkdir(os.path.join(depth_dst_folder,dir))
      try:
        os.stat(os.path.join(normal_dst_folder,dir))
        print('generating depth data in: ', os.path.join(normal_dst_folder,dir))
      except:
        print('creating new depth folder: ', os.path.join(normal_dst_folder,dir))
        os.mkdir(os.path.join(normal_dst_folder,dir))
      depths = []
      normals = []
      scan_path = os.path.join(lidar_path,dir+"/velodyne/")
      idx = 0
      for file in sorted(os.listdir(scan_path)):
        if file == ".DS_Store":
          continue
        bin_pcd = np.fromfile(os.path.join(scan_path,file), dtype = np.float32)
        points = bin_pcd.reshape((-1, 4))[:, 0:3]
        depth_data, proj_vertex = generate_depth(points)
        normal_data = generate_normal(depth_data, proj_vertex)

        # generate the destination path
        depth_dst_path = os.path.join(os.path.join(depth_dst_folder,dir), str(idx).zfill(6))
        normal_dst_path = os.path.join(os.path.join(normal_dst_folder,dir), str(idx).zfill(6))
        
        # save the semantic image as format of .npy
        np.save(depth_dst_path, depth_data)
        np.save(normal_dst_path, normal_data)
        depths.append(depth_data)
        normals.append(normal_data)
        print('finished generating depth data at: ', depth_dst_path)
        print('finished generating intensity data at: ', normal_dst_path)
        idx+=1
    
    return Alldepths,Allnormals




if __name__ == '__main__':
    # Load image, calibration file, label bbox
    lidar_path = "./data/dataset/sequences"
    depth_path = "./data/"
    normal_path = "./data/"
    # poses = load_poses("./data/poses/00.txt")
    depths,normals = generate_embeddings(lidar_path,depth_path,normal_path)
    print("stop")




