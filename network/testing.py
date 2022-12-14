#!/usr/bin/env python3
# Developed by Xieyuanli Chen and Thomas Läbe
# This file is covered by the LICENSE file in the root of this project.
# Brief: Test how good a model performs with a test set.

"""
Test how good a model performs with a test set.

This version calculates the feature volumes == end of a leg of all
input images first and then calls the head for all pairs. This
avoids the multiple loading and calculation of the feature volume for an
image which is member of multiple pairs.

Commandline arguments:
  If no arguments are given, the configuration file 'network.yml' in
  the CURRENT DIRECTORY is used.

  You can also execute this script with

  testing.py <yml-config-file>

  Then the argument should be the yaml configuration file which you intend
  to use.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import yaml
import h5py
from keras.saving import hdf5_format


# To get a log file
import importlib
import logging
importlib.reload(logging) # needed for ipython console

import generateNet
from ImagePairOverlapSequenceFeatureVolume import ImagePairOverlapSequenceFeatureVolume
from ImagePairOverlapOrientationSequence import ImagePairOverlapOrientationSequence
from shared_utils import read_network_config, overlap_orientation_npz_file2string_string_nparray

# ==================== main script ============================================

# Settings (mostly from yaml file)
# --------------------------------
logging.basicConfig(format='%(message)s', level=logging.DEBUG)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

configfilename='config/network.yml'
if len(sys.argv)>1:
    configfilename=sys.argv[1]

config = read_network_config(yaml.load(open(configfilename), yaml.Loader))
    
# Data for testing
# There are three possibilities:
# 1) The standard: Define 'testing_seqs' in the config file. Then,
#    the complete ground truth for all pairs given in the file ground_truth/ground_truth_overlap_yaw.npz
#    is used as test data.
# 2) If no testing sequences are given in the config file, the validation sets 
#    (ground_truth/validation_set.npz) of the training sequences (given in 'training_seqs') are used.
# 3) if no sequences are given (thus neither 'training_seqs' nor 'testing_seqs' appear in configuration file),
#    a single npz file with test data given in parameter 'testdata_npzfile' is used.
if 'testing_seqs' in config:
    logger.info('Using complete ground truth for multiple sequences as test data ...')
    testdata_npzfiles = [config['testing_seqs']]
    testdata_npzfiles = [os.path.join(config['data_root_folder'], seq,
                                      'ground_truth/ground_truth_overlap_yaw.npz') for seq in testdata_npzfiles]
elif 'training_seqs' in config:
    logger.info('Using multiple validation npz files for test data ...')
    training_seqs = config['training_seqs']
    training_seqs = training_seqs.split()

    testdata_npzfiles = [os.path.join(config['data_root_folder'], seq, 'ground_truth/validation_set.npz') for seq in training_seqs]
    
else:
    logger.info('Using a single npz file for test data ...')
    testdata_npzfiles =[config['testdata_npzfile']]
    
# weights from training.
pretrained_weightsfilename=config['pretrained_weightsfilename']

# Path where all experiments are stored. Data of current experiment
# will be in experiment_path/testname
experiments_path=config['experiments_path']

# String which defines experiment
testname=config['testname']
os.makedirs(os.path.join(experiments_path,testname), exist_ok=True)

# Logging should go to experiment directory
fileHandler = logging.FileHandler("{0}/validation_{1}.log".format(os.path.join(experiments_path,testname), 
                                  testname), mode='w')
fileHandler.setFormatter(logging.Formatter(fmt="%(asctime)s %(message)s",
                                           datefmt='%H:%M:%S'))                                  
logger.addHandler(fileHandler)

batch_size = config['batch_size']
no_test_pairs = config['no_test_pairs']

# Create two nets: leg and head
# -----------------------------

network, leg, head = generateNet.generateSiameseNetwork(config['model']['input_shape'], config['model'], False)

logger.info("Created neural net %s for leg   with %d parameters." % 
    ('', leg.count_params()))
logger.info("Created neural net %s for heads with %d parameters." % 
    ('', head.count_params()))

# Load weights from training
if len(pretrained_weightsfilename)>0:
  logger.info("Load old weights from %s" % pretrained_weightsfilename)
  if 'tmp' in pretrained_weightsfilename:
    network.load_weights(pretrained_weightsfilename)
    print("LOADED WEIGHTS")
    pass
  else:
    f = h5py.File(pretrained_weightsfilename)
    hdf5_format.load_weights_from_hdf5_group_by_name(f['model_weights'], network)

# Load Data (only the image filenames)
# ------------------------------------
logger.info("load test data from %s ..." % testdata_npzfiles)
(test_imgf1, test_imgf2, test_dir1, test_dir2, test_overlap, test_orientation) = \
  overlap_orientation_npz_file2string_string_nparray(testdata_npzfiles, shuffle=False)
# Assume same sequence or directory name for all pairs !
test_dir=test_dir1[0]


if no_test_pairs>test_overlap.size:  
  no_test_pairs=test_overlap.size

# Make test set smaller like defined in config file
test_imgf1=test_imgf1[0:no_test_pairs]
test_imgf2=test_imgf2[0:no_test_pairs]
test_overlap=test_overlap[0:no_test_pairs]
test_orientation =test_orientation[0:no_test_pairs]

# Do the validation
# -----------------
logger.info("Test the model:")
logger.info("  batch size is       : %d" % batch_size)
logger.info("  number of test pairs: %d" % no_test_pairs)

# First all feature volumes
logger.info(" ")
logger.info("Compute all feature volumes ...")

# combine pairs: 
# *** This currently assumes we have always the same directory ***
test_allimgs=list(set(test_imgf1) | set(test_imgf2)) 
np_test_allimgs=np.array(test_allimgs)
logger.info("  Number of feature volumes: %d" % np_test_allimgs.size)
# Indizes of a pair in the the feature volumes array
test_pair_idxs=np.zeros((len(test_imgf1),2), dtype='uint')
# use sorted array for faster search
sortidx=np.argsort(np_test_allimgs)
np_test_allimgs_sorted=np_test_allimgs[sortidx]
pos1=sortidx[np.searchsorted(np_test_allimgs_sorted, test_imgf1)]
pos2=sortidx[np.searchsorted(np_test_allimgs_sorted, test_imgf2)]
test_pair_idxs[:,0]=pos1
test_pair_idxs[:,1]=pos2

network_output_size=config['model']['leg_output_width']
    
test_generator_leg = ImagePairOverlapOrientationSequence(
  image_path=config['imgpath'],
  imgfilenames1=test_allimgs,
  imgfilenames2=[],
  dir1=test_dir,
  dir2=[],
  overlap=None,
  orientation=None,
  network_output_size=network_output_size,
  batch_size=batch_size,
  height=config['model']['input_shape'][-3],
  width=config['model']['input_shape'][-2],
  no_channels=config['model']['input_shape'][-1],
  use_depth=config['use_depth'],
  use_normals=config['use_normals'],
  use_class_probabilities=config['use_class_probabilities'],
  use_class_probabilities_pca=config['use_class_probabilities_pca'],
  use_intensity=config['use_intensity'],
)

feature_volumes=leg.predict(test_generator_leg, verbose=1)

# Second head for all pairs
logger.info(" ")
logger.info("Compute head for all %d test pairs ..." % test_pair_idxs.shape[0])

test_generator_head=ImagePairOverlapSequenceFeatureVolume(test_pair_idxs, test_overlap, batch_size, feature_volumes)
model_outputs=head.predict(test_generator_head, verbose=1)
                                     
# Evaluation
# ----------
diffs_overlap=abs(np.squeeze(model_outputs[0])-test_overlap)
mean_diff=np.mean(diffs_overlap)
mean_square_error=np.mean(diffs_overlap*diffs_overlap)
rms_error=np.sqrt(mean_square_error)
max_error=np.max(diffs_overlap)
logger.info(" ")
logger.info("Evaluation overlap on test data:")
logger.info("  Evaluation: mean difference:   %f" % mean_diff)
logger.info("  Evaluation: max  difference:   %f" % max_error)
logger.info("  Evaluation: RMS error        : %f" % rms_error)

# logger.info("Distribution of small values:")
# for max_error in np.arange(0.05, 0.95, 0.05):
#     no_of_elements_smaller=sum(i<=max_error for i in diffs_overlap)
#     logger.info("%6.2f%% of all errors are smaller than %4.2f" % (
#                 100.0*no_of_elements_smaller/diffs_overlap.size, max_error ))
#     if no_of_elements_smaller==diffs_overlap.size:
#         break

logger.info("plot overlap histogram ...")
n_bins=10
plt.figure(1)
plt.clf()
plt.hist(diffs_overlap, bins=n_bins)
plt.xlabel('error in overlap percentage')
plt.ylabel('number of examples')
plt.savefig(os.path.join(experiments_path,testname,'overlap_error_histogram.png'))

network_orientation_output=np.squeeze(np.argmax(model_outputs[1], axis=1))*(360//network_output_size)
# The following takes the circular behaviour of angles into account !
diffs_orientation=np.abs((network_orientation_output - test_orientation + 180) % 360 - 180)
diffs_orientation=diffs_orientation[test_overlap>0.7]
mean_diff=np.mean(diffs_orientation)
mean_square_error=np.mean(diffs_orientation*diffs_orientation)
rms_error=np.sqrt(mean_square_error)
max_error=np.max(diffs_orientation)

logger.info(" ")
logger.info("Evaluation yaw orientation (overlap>0.7) on test data:")
logger.info("  Evaluation: mean difference:   %f" % mean_diff)
logger.info("  Evaluation: max  difference:   %f" % max_error)
logger.info("  Evaluation: RMS error        : %f" % rms_error)

# logger.info("Distribution of small values:")
# for max_error in np.arange(0.05, 0.95, 0.05):
#     no_of_elements_smaller=sum(i<=max_error for i in diffs_orientation)
#     logger.info("%6.2f%% of all errors are smaller than %4.2f" % (
#                 100.0*no_of_elements_smaller/diffs_orientation.size, max_error ))
#     if no_of_elements_smaller==diffs_orientation.size:
#         break


logger.info("plot yaw orientation histogram ...")
n_bins=90
plt.figure(2)
plt.clf()
plt.hist(diffs_orientation, bins=n_bins)
plt.xlabel('error in yaw angle estimation in degrees')
plt.ylabel('number of examples')
plt.savefig(os.path.join(experiments_path,testname,'orientation_error_histogram.png'))

# Save results
# ------------
logger.info("Save results in npz file ...")
test_imgf1 = np.array(test_imgf1)
test_imgf1 = test_imgf1.astype(float)
test_imgf2 = np.array(test_imgf2)
test_imgf2 = test_imgf2.astype(float)

overlapmatrix=np.zeros((len(test_imgf1),4))
overlapmatrix[:,0]=test_imgf1
overlapmatrix[:,1]=test_imgf2
overlapmatrix[:,2]=np.squeeze(model_outputs[0])
overlapmatrix[:,3]=np.squeeze(np.argmax(model_outputs[1], axis=1))

np.savez(os.path.join(experiments_path,testname,"validation_results.npz"), overlapmatrix)

## Show plots
if config['show_plots']:
  print('show plots ...')
  plt.show()
  plt.pause(0.1)

logger.info("... done.")
