import yaml
import os
import sys
import numpy as np

import h5py
from keras.saving import hdf5_format

import generateNet
from ImagePairOverlapSequenceFeatureVolume import ImagePairOverlapSequenceFeatureVolume
from ImagePairOverlapOrientationSequence import ImagePairOverlapOrientationSequence

from shared_utils import read_network_config

class Infer():
  """ A class used for inferring overlap and yaw-angle between LiDAR scans.
  """
  
  def __init__(self, config):
    """ Initialization of the class
        Args:
          config: A dict with configuration values, usually loaded from a yaml file
    """

    self.config = read_network_config(config)

    self.datasetpath = self.config['data_root_folder']
    self.seq = self.config['infer_seqs']
    
    self.input_shape = self.config['model']['input_shape']
    self.batch_size = self.config['batch_size']
    
    self.network, self.leg, self.head = generateNet.generateSiameseNetwork(self.input_shape, self.config['model'], False)

    # previous feature volumes
    self.feature_volumes = {}
    
    # Load weights from training
    pretrained_weightsfilename = self.config['pretrained_weightsfilename']
    if len(pretrained_weightsfilename) > 0:
      f = h5py.File(pretrained_weightsfilename)
      hdf5_format.load_weights_from_hdf5_group_by_name(f['model_weights'], self.network)
    else:
      print('Pre-trained weights was not found in:', pretrained_weightsfilename)


  def infer_one(self, filepath1, filepath2):
    """ Infer with one input pair.
          Args:
            filepath1: path of LiDAR scan 1
            filepath2: path of LiDAR scan 2
            
          Returns:
            [overlap, yaw]
    """
    # check file format
    if not filepath1.endswith('.bin') or not filepath2.endswith('.bin'):
      raise Exception('Please check the LiDAR file format, '
                      'this implementation currently only works with .bin files.')
    
    filename1 = os.path.basename(filepath1).replace('.bin', '')
    filename2 = os.path.basename(filepath2).replace('.bin', '')
    self.filenames = np.array([filename1, filename2])
    
    # check preprocessed data
    preprocess_data_folder = os.path.join(self.datasetpath, self.seq)
    if not os.path.isdir(preprocess_data_folder):
      raise Exception('Please first generate preprocessed input data.')
    
    test_generator_head = self.get_generator([filename1,],[filename2,])
    model_outputs = self.network.predict(test_generator_head, verbose=1)

    overlap_out = model_outputs[0][0]
    yaw_out = 180 - np.argmax(model_outputs[1], axis=1)
    
    return overlap_out, yaw_out


  def infer_multiple(self, current_frame_id, reference_frame_id):
    """ Infer for loopclosing: The current frame versus old frames.
        This is a special function, because only the feature volume of the current frame
        is computed. For the older reference frames the feature volumes must be already
        there. This is usually the case, because they were the "current frame" in
        previous calls of the function and the feature volumes are stored within
        this class for every call. For the starting frame, call this function
        with an empty list of reference_frame_id.
        
        For a more general usage use Infer.infer_multiple_vs_multiple().
    
        Args:
          current_frame_id: The id (an int) of the current frame. This corresponds
                            to depth and normal and scan files, 
                            e.g. 6 --> file 000006.bin or 000006.npy is used.
                            For this frame the feature volume is calculated and appended to
                            the list of already calculated feature volumes.
          reference_frame_id: a list of ids (aka a list of ints) of previous frames. 
                              Can be empty
        Returns:
          A tuple (overlaps, yaws) with two lists of the overlaps and yaw angles between the scans
    """

    filename = [str(current_frame_id).zfill(6)]
    self.feature_volumes[current_frame_id] = self.create_feature_volumes(filename)[0]
    
    if len(reference_frame_id) > 0:
      pair_indices = np.zeros((len(reference_frame_id), 2), dtype=int)
      pair_indices[:, 1] = current_frame_id
      pair_indices[:, 0] = reference_frame_id
      
      test_generator_head = ImagePairOverlapSequenceFeatureVolume(pair_indices, None, self.batch_size, self.feature_volumes)
      model_outputs = self.head.predict(test_generator_head, verbose=1)
      
      overlap_out = model_outputs[0].squeeze()
      yaw_out = 180 - np.argmax(model_outputs[1], axis=1)
    
      return overlap_out, yaw_out
    
    else:
      return None


  def get_generator(self, filenames1, filenames2=[]):

    return ImagePairOverlapOrientationSequence(
      image_path=self.datasetpath,
      imgfilenames1=filenames1,
      imgfilenames2=filenames2,
      dir1=self.seq,
      dir2=self.seq,
      overlap=None,
      orientation=None,
      network_output_size=self.config['model']['leg_output_width'],
      batch_size=self.batch_size,
      height=self.input_shape[-3],
      width=self.input_shape[-2],
      no_channels=self.input_shape[-1],
      use_depth=self.config['use_depth'],
      use_normals=self.config['use_normals'],
      use_class_probabilities=self.config['use_class_probabilities'],
      use_class_probabilities_pca=self.config['use_class_probabilities_pca'],
      use_intensity=self.config['use_intensity'],
    )


  def create_feature_volumes(self, filenames):
    """ create feature volumes, thus execute the leg.
        Args:
          filenames: numpy array of input file names (list of strings without extension, e.g. ['000000', '000001'])
        Returns:
          A n x width x height x channels numpy array of feature volumes
    """
    
    return self.leg.predict(self.get_generator(filenames), verbose=1)

        
# Test the infer functions
if __name__ == '__main__':
  configfilename = 'config/network.yml'
  
  if len(sys.argv) > 1:
    configfilename = sys.argv[1]
  if len(sys.argv) > 2:
    scan1 = sys.argv[2]
    scan2 = sys.argv[3]
  else:
    scan1 = '000020.bin'
    scan2 = '000021.bin'
  
  config = yaml.load(open(configfilename))
  infer = Infer(config)
  
  print('Test infer one ...')
  overlap, yaw = infer.infer_one(scan1, scan2)
  print("Overlap:  ", overlap)
  print("Orientation:  ", yaw)
  
  print('Test infer multiple (last scan vs previous) ...')
  # Note that this is special for loop-closing, all previous have to be asked !
  infer.infer_multiple(0, [])
  overlaps, yaws = infer.infer_multiple(1, [0])
  overlaps, yaws = infer.infer_multiple(2, [0,1])
  print("Overlaps:  ", overlaps)
  print("Orientations:  ", yaws)
