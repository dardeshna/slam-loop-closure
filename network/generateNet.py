import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Dense, Flatten, Reshape, Lambda, Layer
from keras.models import Model
from keras.regularizers import l2


class NormalizedCorrelation2D(Layer):

  def call(self, inputs):

    input_1, input_2 = inputs

    # shape = (batch_size, height, width, channels)
    width = input_1.shape[-2]
    n_pad = width // 2

    # circular padding of first input -> (batch_size, height, 2*width-1, channels)
    input_1 = K.concatenate([input_1[..., n_pad:, :], input_1, input_1[..., :n_pad-1, :]], axis=-2)

    # sliding window over first input -> (batch_size, height, n_windows=width, width, channels)
    input_1 = tf.signal.frame(input_1, frame_length=width, frame_step=1, axis=-2)

    # add additional dimension to second input -> (batch_size, height, 1, width, channels)
    input_2 = K.expand_dims(input_2, -3)

    # multiply -> (batch_size, height, n_windows, width, channels)
    product = input_1 * input_2

    # sum original axes -> (batch_size, n_windows)
    return K.sum(product, axis=[-4,-2,-1])


def DeltaLayer(encoded_l, encoded_r):
  """
  A Layer which computes all possible absolute differences of
  all pixels. Input are two feature volumes, e.g. result of a conv layer
  Hints:
  - The Reshape reshapes a matrix row-wise, that means,

    Reshape( (6,1) ) ([ 1 2 3
                      4 5 6]) is

                      1
                      2
                      3
                      4
                      5
                      6
  - Algorithm:
    - The left  leg is reshaped to a w*h x 1  column vector (for each channel)
    - The right leg is reshaped to a  1 x w*h row vector (for each channel)
    - The left is tiled along colum axis, so from w*h x 1 to w*h x w*h (per channel)
    - The right is tiled along row axis, so from 1 x w*h to w*h x w*h
    - The absolute difference is calculated
  Args:
      encoded_l, encoded_r : left and right image tensor (batchsize,w,h,channels)
                             must have same size
  Returns:
      difference tensor, has size (batchsize, w*h, w*h, channels)
  """
  w = encoded_l.shape[1]
  h = encoded_l.shape[2]
  chan = encoded_l.shape[3]
  reshapel = Reshape((w * h, 1, chan))
  reshaped_l = reshapel(encoded_l)
  reshaper = Reshape((1, w * h, chan))
  reshaped_r = reshaper(encoded_r)
  
  tiled_l = Lambda(lambda x: K.tile(x, [1, 1, w * h, 1]))(reshaped_l)
  tiled_r = Lambda(lambda x: K.tile(x, [1, w * h, 1, 1]))(reshaped_r)
  
  diff = Lambda(lambda x: K.abs(x[0] - x[1]))([tiled_l, tiled_r])
  
  return diff

def generateDeltaLayerConv1NetworkHead(encoded_l, encoded_r, config={}):
  """
  Generate Head of DeltaLayerConv1Network.
  Args:
    encoded_l, encoded_r: the feature volumes of the two images,
                          thus the last tensor of the leg
    config: dictionary of configuration parameters, usually from a yaml file
            All keys have default arguments, so they need not to be present
            
  Returns:
    the final tensor of the head which is 1x1, the overlap percentage
    0.0-1.0
  """

  # combine the two legs
  diff = DeltaLayer(encoded_l, encoded_r)

  # densify the information across feature maps

  default_kwargs = {
    'activation' : 'relu',
    'kernel_regularizer': None,
  }

  i = 1
  for layer in config['head_architecture']:
    if 'name' not in layer:
      layer['name'] = f'c_conv{i}'
      i += 1
    
    kwargs = {**default_kwargs, **layer}
    conv = Conv2D(**kwargs)
    diff = conv(diff)

  flattened = Flatten()(diff)

  prediction = Dense(1, activation='sigmoid', name='overlap_output')(flattened)

  return prediction


def generate360OutputkLegs(left_input, right_input, config={},
                           smallNet=False, trainable=True):
  """
  Generate legs like in the DeltaLayerConv1Network.
  Here we use several Conv2D layer to resize the output of leg into 360

  Args:
    left_input, right_input: Two tensors of size input_shape which define the input
                             of the two legs
    config: dictionary of configuration parameters, usually from a yaml file
            All keys have default arguments, so they need not to be present
    smallNet: a boolean. If true, a very tiny net is defined. Default: False

  Returns:
    a tuple with two tensors: the left and right feature volume
  """

  # build convnet to use in each siamese 'leg'
  if (smallNet):
    finalconv = Conv2D(2, (5, 15), activation='relu',
                       padding='valid', strides=5,
                       name='s_conv1', kernel_regularizer=l2(2e-4), trainable=trainable)
    l = finalconv(left_input)
    r = finalconv(right_input)
    return (l, r)
    
  else:
  
    default_kwargs = {
      'activation' : 'relu',
      'kernel_regularizer': None, # l2(1e-8)
      'trainable': trainable,
    }

    l, r = left_input, right_input

    i = 1
    for layer in config['leg_architecture']:
      if 'name' not in layer:
        layer['name'] = f's_conv{i}'
        i += 1
      
      kwargs = {**default_kwargs, **layer}
      conv = Conv2D(**kwargs)
      l, r = conv(l), conv(r)
  
    return (l, r)

  
def generateCorrelationHead(encoded_l, encoded_r, config={}):
    """
      Generate a head which does correlation.
      
    Args:
      encoded_l, encoded_r: the feature volumes of the two images,
                            thus the last tensor of the leg. Must be of size
                            batch_idx x 1 x no_col x no_channels
  
      config: dictionary of configuration parameters, usually from a yaml file
              Current parameters used here:
    
  
    Returns:
      the output feature (volume) of the head. Here: A feature volume with size 1XMx1
    """
    
    norm_corr = NormalizedCorrelation2D(name='orientation_output')([encoded_l, encoded_r])
    
    return norm_corr


def generateSiameseNetwork(input_shape, config={}, smallNet=False):
  """
  Generate a siamese network for overlap detection. Which legs and which
  head is used will be given in the config parameter.
  
  Args:
    input_shape: A tupel with three elements which is the size of the input images.
    config: dictionary of configuration parameters, usually from a yaml file
            Current parameters used here:
            legsType: name of the function (without "generate") for the legs
            headType: name of the function (without "generate") for the heads
            The config is given to the head and legs, so additional parameters can
            be given.

  Returns:
    the neural net as a keras.model
  """
  
  # Define the input
  left_input = Input(input_shape)
  right_input = Input(input_shape)
  
  # The two legs
  (encoded_l, encoded_r) = generate360OutputkLegs(left_input, right_input, config, smallNet)

  # Independent leg network
  leg = Model(inputs=left_input, outputs=encoded_l, name='leg')
  
  # The overlap head
  prediction_overlap = generateDeltaLayerConv1NetworkHead(encoded_l, encoded_r, config)
  
  # The orientation head
  prediction_orientation = generateCorrelationHead(encoded_l, encoded_r, config)

  # Independent head network
  head = Model(inputs=[encoded_l, encoded_r], outputs=[prediction_overlap, prediction_orientation], name='head')
  
  # Generate a keras model out of the input and output tensors
  siamese_net = Model(inputs=[left_input, right_input], outputs=[prediction_overlap, prediction_orientation], name='siamese_net')

  return siamese_net, leg, head


# For testing/debuging
if __name__ == "__main__":
  input_shape = (64, 900, 16)
  config = {}
  config['legsType'] = '360OutputkLegs'
  config['overlap_head'] = 'DeltaLayerConv1NetworkHead'
  config['orientation_head'] = 'CorrelationHead'
  
  config['additional_unsymmetric_layer3a'] = True
  config['strides_layer1'] = [2, 2]
  model, _, _ = generateSiameseNetwork(input_shape, config)
  model.summary()
