# Networks used in vqvae1

import tensorflow as tf
import numpy as np

# Get/create weight tensor for a convolutional or fully-connected layer
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal()) * wscale
    else:
        return tf.get_variable('weight', shape=shape, initializer=tf.initializers.random_normal(0, std))

# Convolutional layer
def conv2d(x, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([kernel, kernel, x.shape[-1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, filter=w, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')

# Convolutional transpose layer
def conv2d_transpose(x, size, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([kernel, kernel, fmaps, x.shape[-1].value], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    input_shape = tf.shape(x)
    output_shape = [input_shape[0], int(size), int(size), int(fmaps)]
    return tf.nn.conv2d_transpose(x, filter=w, output_shape=output_shape, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')

# Apply bias to the given activation tensor
def apply_bias(x):
    b = tf.get_variable('bias', shape=[x.shape[-1]], initializer=tf.initializers.zeros())
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + tf.reshape(b, [1, -1])
    else:
        return x + tf.reshape(b, [1, 1, 1, -1])

# Residual stack
def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
	for i in range(num_residual_layers):
		h_i = tf.nn.relu(h)

		with tf.variable_scope('res3x3_%d' % i):
			h_i = conv2d(h_i, fmaps=num_residual_hiddens, kernel=3, strides=1)
			h_i = apply_bias(h_i)
			h_i = tf.nn.relu(h_i)

		with tf.variable_scope('res1x1_%d' % i):
			h_i = conv2d(h_i, fmaps=num_hiddens, kernel=1, strides=1)
			h_i = apply_bias(h_i)

		h += h_i
	return h

# Encoder networks
def encoder(x, num_hiddens, num_residual_layers, num_residual_hiddens):
  with tf.variable_scope('Encoder_Conv_0', reuse=tf.AUTO_REUSE):
    h = conv2d(x, fmaps=num_hiddens/2, kernel=4, strides=2)
    h = apply_bias(h)
    h = tf.nn.relu(h)
  
  with tf.variable_scope('Encoder_Conv_1', reuse=tf.AUTO_REUSE):
    h = conv2d(h, fmaps=num_hiddens, kernel=4, strides=2)
    h = apply_bias(h)
    h = tf.nn.relu(h)
  
  with tf.variable_scope('Encoder_Conv_2', reuse=tf.AUTO_REUSE):
    h = conv2d(h, fmaps=num_hiddens, kernel=3, strides=1)
    h = apply_bias(h)

  with tf.variable_scope('Encoder_residual_stack', reuse=tf.AUTO_REUSE):
    h = residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens)
  return h # shape=[?, image_size/4, image_size/4, num_hiddens]

# Decoder networks
def decoder(x, num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel):
  with tf.variable_scope('Decoder_Conv_0', reuse=tf.AUTO_REUSE):
    h = conv2d(x, fmaps=num_hiddens, kernel=3, strides=1)
    h = apply_bias(h)

  with tf.variable_scope('Decoder_residual_stack', reuse=tf.AUTO_REUSE):
    h = residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens)
    h = tf.nn.relu(h)

  with tf.variable_scope('Decoder_Conv_1', reuse=tf.AUTO_REUSE):
    h = conv2d_transpose(h, size=image_size/2, fmaps=num_hiddens/2, kernel=4, strides=2)
    h = apply_bias(h)
    h = tf.nn.relu(h)

  with tf.variable_scope('Decoder_Conv_2', reuse=tf.AUTO_REUSE):
    x_recon = conv2d_transpose(h, size=image_size, fmaps=num_channel, kernel=4, strides=2)
    x_recon = apply_bias(x_recon)
    x_recon = tf.nn.sigmoid(x_recon) - 0.5
  return x_recon

