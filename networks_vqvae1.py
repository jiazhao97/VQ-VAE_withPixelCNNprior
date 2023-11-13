# Networks used in vqvae1
import tensorflow as tf
import numpy as np
# Get/create weight tensor for a convolutional or fully-connected layer
def get_weight(shape, gain=np.sqrt(2), use_wscale=False, fan_in=None):
    if fan_in is None: fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in) # He init
    if use_wscale:
        wscale = tf.constant(np.float32(std), name='wscale')
        return tf.Variable(initial_value=tf.random.normal(shape, stddev=std), name='weight') * wscale
    else:
        return tf.Variable(initial_value=tf.random.normal(shape, stddev=std), name='weight')
# Convolutional layer
def conv2d(x, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([kernel, kernel, x.shape[-1], fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.nn.conv2d(x, filters=w, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')
# Convolutional transpose layer
def conv2d_transpose(x, size, fmaps, kernel, strides, gain=np.sqrt(2), use_wscale=False):
    w = get_weight([kernel, kernel, fmaps, x.shape[-1]], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    input_shape = tf.shape(x)
    output_shape = [input_shape[0], int(size), int(size), int(fmaps)]
    return tf.nn.conv2d_transpose(x, filters=w, output_shape=output_shape, strides=[1, strides, strides, 1], padding='SAME', data_format='NHWC')

# Apply bias to the given activation tensor
def apply_bias(x):
    b = tf.Variable(initial_value=tf.zeros(shape=[x.shape[-1]]), name='bias')
    b = tf.cast(b, x.dtype)
    if len(x.shape) == 2:
        return x + tf.reshape(b, [1, -1])
    else:
        return x + tf.reshape(b, [1, 1, 1, -1])

# Residual stack
def residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens):
    for i in range(num_residual_layers):
        h_i = tf.nn.relu(h)
        h_i = tf.keras.layers.Conv2D(filters=num_residual_hiddens, kernel_size=3, strides=1, padding='same')(h_i)
        h_i = apply_bias(h_i)
        h_i = tf.nn.relu(h_i)
        h_i = tf.keras.layers.Conv2D(filters=num_hiddens, kernel_size=1, strides=1, padding='same')(h_i)
        h_i = apply_bias(h_i)
        h += h_i
    return h


# Encoder networks
# Encoder networks
def encoder(x, num_hiddens, num_residual_layers, num_residual_hiddens):
    h = tf.keras.layers.Conv2D(filters=num_hiddens/2, kernel_size=4, strides=2, padding='same')(x)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.activations.relu(h)
  
    h = tf.keras.layers.Conv2D(filters=num_hiddens, kernel_size=4, strides=2, padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.activations.relu(h)
  
    h = tf.keras.layers.Conv2D(filters=num_hiddens, kernel_size=3, strides=1, padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens)
    return h # shape=[?, image_size/4, image_size/4, num_hiddens]

# Decoder networks
def decoder(x, num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel):
    h = tf.keras.layers.Conv2D(filters=num_hiddens, kernel_size=3, strides=1, padding='same')(x)
    h = tf.keras.layers.BatchNormalization()(h)
    h = residual_stack(h, num_hiddens, num_residual_layers, num_residual_hiddens)
    h = tf.keras.activations.relu(h)
  
    h = tf.keras.layers.Conv2DTranspose(filters=num_hiddens/2, kernel_size=4, strides=2, padding='same')(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.activations.relu(h)
  
    x_recon = tf.keras.layers.Conv2DTranspose(filters=num_channel, kernel_size=4, strides=2, padding='same')(h)
    x_recon = tf.keras.layers.BatchNormalization()(x_recon)
    x_recon = tf.keras.activations.sigmoid(x_recon) - 0.5
    return x_recon
