import numpy as np


def read_network_config(config):

    default_config = {
        'use_depth': True,
        'use_normals': True,
        'use_class_probabilities': False,
        'use_class_probabilities_pca': False,
        'use_intensity': False,
        'rotate_training_data': 0,
        'min_overlap_for_angle': 0.7,
        'data_root_folder': '',
        'pretrained_weightsfilename': '',
    }

    config = {**default_config, **config}

    # Image Path: Use path from yml or data_root_folder if not given
    if 'imgpath' not in config:
        config['imgpath'] = config['data_root_folder']
        
    # number of channels for input
    input_channels = 0
    if config['use_depth']:
      input_channels += 1
    if config['use_normals']:
      input_channels += 3
    if config['use_intensity']:
      input_channels += 1
    if config['use_class_probabilities']:
      if config['use_class_probabilities_pca']:
        input_channels += 3
      else:
        input_channels += 20
    
    # input shape of model
    input_shape = config['model']['input_shape']
    if len(input_shape) >= 3:
      input_shape[-1] = input_channels
    elif len(input_shape) == 2:
      input_shape.append(input_channels)

    # update config with computed information
    config['model']['input_shape'] = input_shape
    config['model']['input_channels'] = input_channels
    config['model']['input_width'] = input_shape[-2]
    config['model']['input_height'] = input_shape[-3]
    
    return config


def overlap_orientation_npz_file2string_string_nparray(npzfilenames, shuffle=True):
    """ Load overlap files as npz array and convert it to a tuple.
        Args:
          npzfilenames: list of npz filenames
          shuffle: Boolean, define whether to shuffle.
        Returns:
          A tupple (imgf1, imgf2, dir1, dir2, overlap) with
            imgf1: first partner of a pair, list of  n strings, formatted as the filenames (%06d)
            imgf1: second partner of a pair, list of n strings, formatted as the filenames (%06d)
            dir1 : n top directories of the dataset for imgf1 as list. This is the directory name of the sequence,
                   e.g. dir1[0]/depth_map/imgf1[0].png
            dir2 : n top directories of the dataset for imgf2 as list. This is the directory name of the sequence,
                   e.g. dir2[0]/depth_map/imgf2[0].png
            overlap: numpy array of size (n,)

       If no directory information is available in the npz (old format, only one
       array in the npz file), then dir1 and dir2 are all empty strings.
       
       The overlap and image file number are assumed to be in the second nx3 float array of
       the npz, whereas the directory names are in the first array nx2 (string) of the
       npz file.
       
       The data will be shuffled after loading if shuffle=True, thus the sequence of pairs will be 
       changed randomly.
    """
    imgf1_all = []
    imgf2_all = []
    dir1_all = []
    dir2_all = []
    overlap_all = []
    orientation_all = []
    
    for npzfilename in npzfilenames:
        h=np.load(npzfilename, allow_pickle=True)
        
        if len(h.files)==1:
            # old format
            imgf1=np.char.mod('%06d',h[h.files[0]][:,0]).tolist()
            imgf2=np.char.mod('%06d',h[h.files[0]][:,1]).tolist()
            overlap=h[h.files[0]][:,2]
            orientation = h[h.files[0]][:,3]
            n=len(imgf1)
            dir1=np.array(['' for _ in range(n)]).tolist()
            dir2=np.array(['' for _ in range(n)]).tolist()
        else:
            imgf1=np.char.mod('%06d',h['overlaps'][:,0]).tolist()
            imgf2=np.char.mod('%06d',h['overlaps'][:,1]).tolist()
            overlap=h['overlaps'][:,2]
            orientation = h['overlaps'][:,3]
            dir1=(h['seq'][:,0]).tolist()
            dir2=(h['seq'][:,1]).tolist()
            
        if shuffle:
            shuffled_idx=np.random.permutation(overlap.shape[0])
            imgf1=(np.array(imgf1)[shuffled_idx]).tolist()
            imgf2=(np.array(imgf2)[shuffled_idx]).tolist()
            dir1=(np.array(dir1)[shuffled_idx]).tolist()
            dir2=(np.array(dir2)[shuffled_idx]).tolist()
            overlap=overlap[shuffled_idx]
            orientation=orientation[shuffled_idx]

        imgf1_all.extend(imgf1)
        imgf2_all.extend(imgf2)
        dir1_all.extend(dir1)
        dir2_all.extend(dir2)
        overlap_all.extend(overlap)
        orientation_all.extend(orientation)
    
    return (imgf1_all, imgf2_all, dir1_all, dir2_all, np.asarray(overlap_all), np.asarray(orientation_all))