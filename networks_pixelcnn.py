# Networks used in Pixel CNN
# Networks used in Pixel CNN
import tensorflow as tf
import numpy as np
def get_weights_pixelcnn(shape, name, horizontal=False, mask=None):
    '''
    shape: [filter_size, filter_size, img_channel, f_map] (filter_size should be odd)
    horizontal: True for verticak stack; False for horizontal stack
    mask: None for the filter with no mask; 
          'a' for masked filter in the first layer;
          'b' for masked filter in the second, third,... layers.
    '''
    weights_initializer = tf.keras.initializers.GlorotNormal()
    W = tf.Variable(initial_value=weights_initializer(shape=shape), name=name, dtype=tf.float32)
    if mask:
        filter_mid_y = shape[0]//2
        filter_mid_x = shape[1]//2 # [filter_mid_y, filter_mid_x] is the center of the filter
        mask_filter = np.ones(shape=shape, dtype=np.float32)
        if horizontal: 
            mask_filter[filter_mid_y+1:, :, :, :] = 0.0
            mask_filter[filter_mid_y, filter_mid_x+1:, :, :] = 0.0
            if mask == 'a':
                mask_filter[filter_mid_y, filter_mid_x, :, :] = 0.0
        else: 
            if mask == 'a':
                mask_filter[filter_mid_y:, :, :, :] = 0.0
            else:
                mask_filter[filter_mid_y+1:, :, :, :] = 0.0
        W.assign(W * mask_filter)
    return W
def gated_conv_pixelcnn(W_shape_f, fan_in, horizontal, payload=None, mask=None):
    '''
    W_shape_f: [filter_size, filter_size, f_map] (filter_size should be odd)
    fan_in: data to fit in
    '''
    in_dim = fan_in.shape[-1]
    W_shape = [W_shape_f[0], W_shape_f[1], in_dim, W_shape_f[2]]
    b_shape = W_shape[2]
    W_f = get_weights_pixelcnn(shape=W_shape, name="v_W", horizontal=horizontal, mask=mask)
    W_g = get_weights_pixelcnn(shape=W_shape, name="h_W", horizontal=horizontal, mask=mask)
    b_f_total = tf.Variable(initial_value=tf.zeros(shape=b_shape), name="v_b", dtype=tf.float32)
    b_g_total = tf.Variable(initial_value=tf.zeros(shape=b_shape), name="h_b", dtype=tf.float32)
    conv_f = tf.nn.conv2d(input=fan_in, filters=W_f, strides=[1,1,1,1], padding='SAME')

def simple_conv_pixelcnn(W_shape_f, fan_in, activation=True):
    '''
    W_shape_f: [filter_size, filter_size, f_map] (filter_size should be odd)
    fan_in: data to fit in
    '''
    in_dim = fan_in.shape[-1]
    W_shape = [W_shape_f[0], W_shape_f[1], in_dim, W_shape_f[2]]
    b_shape = W_shape_f[2]
    W = get_weights_pixelcnn(shape=W_shape, name="W")
    b = tf.Variable(initial_value=tf.zeros(shape=b_shape), name="b", dtype=tf.float32)
    conv = tf.nn.conv2d(input=fan_in, filters=W, strides=[1,1,1,1], padding='SAME')
    if activation:
        fan_out = tf.nn.relu(tf.add(conv, b))
    else:
        fan_out = tf.add(conv, b)
    return fan_out

import tensorflow as tf
from tensorflow.keras import layers

def pixelcnn(inputs, num_layers_pixelcnn, fmaps_pixelcnn, num_embeddings, code_size):
    inputs_shape = tf.shape(inputs)
    inputs = tf.reshape(inputs, [inputs_shape[0], code_size, code_size, 1])
    inputs = tf.cast(inputs, tf.float32)

    v_stack_in = inputs
    h_stack_in = inputs

    for i in range(num_layers_pixelcnn):
        filter_size = 3 if i > 0 else 7
        mask = 'b' if i > 0 else 'a'
        residual = True if i > 0 else False
        i = str(i)

        v_stack = gated_conv_pixelcnn(W_shape_f=[filter_size, filter_size, fmaps_pixelcnn], fan_in=v_stack_in, horizontal=False, mask=mask)
        v_stack_in = v_stack

        v_stack_1 = simple_conv_pixelcnn(W_shape_f=[1, 1, fmaps_pixelcnn], fan_in=v_stack_in)

        h_stack = gated_conv_pixelcnn(W_shape_f=[filter_size, filter_size, fmaps_pixelcnn], fan_in=h_stack_in, horizontal=True, payload=v_stack_1, mask=mask)

        h_stack_1 = simple_conv_pixelcnn(W_shape_f=[1, 1, fmaps_pixelcnn], fan_in=h_stack)
        if residual:
            h_stack_1 += h_stack_in
        h_stack_in = h_stack_1

    fc1 = simple_conv_pixelcnn(W_shape_f=[1, 1, fmaps_pixelcnn], fan_in=h_stack_in)
    fc2 = simple_conv_pixelcnn(W_shape_f=[1, 1, num_embeddings], fan_in=fc1, activation=False)

    dist = tf.random.categorical(logits=fc2, num_samples=1)
    sampled_pixelcnn = dist.sample()
    log_prob_pixelcnn = dist.log_prob(sampled_pixelcnn)

    inputs = tf.reshape(inputs, tf.shape(inputs)[:-1])
    inputs = tf.cast(inputs, tf.int32)
    loss_per_batch = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc2, labels=inputs), axis=[1,2])
    loss_pixelcnn = tf.reduce_mean(loss_per_batch)

    return {'loss_pixelcnn': loss_pixelcnn,
            'sampled_pixelcnn': sampled_pixelcnn,
            'log_prob_pixelcnn': log_prob_pixelcnn}
