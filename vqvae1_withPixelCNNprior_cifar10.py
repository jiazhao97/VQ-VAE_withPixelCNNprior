import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
from six.moves import cPickle

import networks_vqvae1 as vqvae_nets
import vector_quantizer as vq
import networks_pixelcnn as pixelcnn_nets

#--------------------------------------------------------------------------
# Set hyper-parameters
local_data_dir = "/home/jzhaoaz/jiazhao/DataSets/cifar10/cifar-10-python/cifar-10-batches-py" # gpu3
# local_data_dir = "/home/share/Dataset/cifar10/cifar-10-batches-py" # gpu4

save_res_dir = "res_vqvae1_PixelCNN_cifar10_K8_D64_gradclip_n20000_lr1e_3_batch100"
if not os.path.exists(save_res_dir):
    os.makedirs(save_res_dir)

# for vqvae1
num_training_updates = 20000
image_size = 32
num_channel = 3
batch_size = 100

num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64 # D
num_embeddings = 8 # K
commitment_cost = 0.25

learning_rate = 3e-4

# for PixelCNN
num_training_updates_pixelcnn = 30000

num_layers_pixelcnn = 12
fmaps_pixelcnn = 32
code_size = 8

learning_rate_pixelcnn = 1e-3
grad_clip_pixelcnn = 1.0

#--------------------------------------------------------------------------
# Placeholder
x = tf.placeholder(tf.float32, shape=(None, image_size, image_size, num_channel))
data_pixelcnn = tf.placeholder(tf.int32, shape=(None, code_size, code_size)) # Train
sampled_code_pixelcnn = tf.placeholder(tf.int32, shape=(None, code_size, code_size)) # Plot

#--------------------------------------------------------------------------
# Data

# Tools to convert images to floating point with the range [-0.5, 0.5]
def cast_and_normalise_images(images):
    images = (tf.cast(images, tf.float32) / 255.0) - 0.5
    return images

# Tools to reconstruct data
def convert_batch_to_image_grid(images, image_size=image_size, num_channel=num_channel):
    reshaped = (images.reshape(10, 10, image_size, image_size, num_channel)
                .transpose(0, 2, 1, 3, 4)
                .reshape(10 * image_size, 10 * image_size, num_channel))
    return reshaped + 0.5

# Tools to load data
def unpickle(filename):
  with open(filename, 'rb') as fo:
    return cPickle.load(fo, encoding='latin1')
  
def reshape_flattened_image_batch(flat_image_batch):
  return flat_image_batch.reshape(-1, 3, 32, 32).transpose([0, 2, 3, 1])  # convert from NCHW to NHWC

def combine_batches(batch_list):
  images = np.vstack([reshape_flattened_image_batch(batch['data'])
                      for batch in batch_list])
  labels = np.vstack([np.array(batch['labels']) for batch in batch_list]).reshape(-1, 1)
  return {'images': images, 'labels': labels}

# Load data
train_data_dict = combine_batches([
  unpickle(os.path.join(local_data_dir, 'data_batch_%d' % i))
    for i in range(1,6)
    ])
test_data_dict  = combine_batches([unpickle(os.path.join(local_data_dir, 'test_batch'))])

data_variance = np.var(train_data_dict['images'] / 255.0)

train_dataset = (tf.data.Dataset.from_tensor_slices(train_data_dict['images'])
        .map(cast_and_normalise_images)
        .shuffle(10000)
        .repeat()
        .batch(batch_size))
test_dataset  = (tf.data.Dataset.from_tensor_slices(test_data_dict['images'])
        .map(cast_and_normalise_images)
        .repeat()
        .batch(batch_size))
train_iterator = train_dataset.make_one_shot_iterator()
test_iterator  = test_dataset.make_one_shot_iterator()
train_dataset_batch_tf = train_iterator.get_next()
test_dataset_batch_tf  = test_iterator.get_next()
print("Data loading is finished...")


#--------------------------------------------------------------------------
# Training process

#--- Train process - vqvae1
z = vqvae_nets.encoder(x, num_hiddens, num_residual_layers, num_residual_hiddens)
with tf.variable_scope('to_vq'):
  z = vqvae_nets.conv2d(z, fmaps=embedding_dim, kernel=1, strides=1)
vq_output = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost)
x_recon = vqvae_nets.decoder(vq_output["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)

recon_error = tf.reduce_mean((x_recon - x)**2) / data_variance  # Normalized MSE
loss = recon_error + vq_output["loss"]
perplexity = vq_output["perplexity"] 

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

#--- Train process - PixelCNN for priors
train_databatch_pixelcnn = vq_output["encoding_indices"]

pixelcnn_output = pixelcnn_nets.pixelcnn(data_pixelcnn, num_layers_pixelcnn, fmaps_pixelcnn, num_embeddings, code_size)
loss_pixelcnn = pixelcnn_output["loss_pixelcnn"]
sampled_pixelcnn_train = pixelcnn_output["sampled_pixelcnn"]

# optimizer_pixelcnn = tf.train.RMSPropOptimizer(learning_rate_pixelcnn).minimize(loss_pixelcnn)
trainer_pixelcnn = tf.train.RMSPropOptimizer(learning_rate=learning_rate_pixelcnn)
gradients_pixelcnn = trainer_pixelcnn.compute_gradients(loss_pixelcnn)
clipped_gradients_pixelcnn = map(lambda gv: gv if gv[0] is None else [tf.clip_by_value(gv[0], -grad_clip_pixelcnn, grad_clip_pixelcnn), gv[1]], gradients_pixelcnn)
# clipped_gradients_pixelcnn = [(tf.clip_by_value(_[0], -grad_clip_pixelcnn, grad_clip_pixelcnn), _[1]) for _ in gradients_pixelcnn]
optimizer_pixelcnn = trainer_pixelcnn.apply_gradients(clipped_gradients_pixelcnn)

#--- For plots
vq_output_pixelcnn = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost, only_lookup=True, inputs_indices=sampled_code_pixelcnn)
x_recon_pixelcnn = vqvae_nets.decoder(vq_output_pixelcnn["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)

test_vq_output = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost, random_gen=True)
test_recon = vqvae_nets.decoder(test_vq_output["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
test_recon_rand = vqvae_nets.decoder(test_vq_output["rand_quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
test_recon_near = vqvae_nets.decoder(test_vq_output["near_quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)


#--------------------------------------------------------------------------
# Train

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

#--- Train vqvae1
train_res_recon_error = []
train_res_perplexity = []
train_res_loss = []
for i in range(num_training_updates):
  train_data_batch = sess.run(train_dataset_batch_tf)
  train_feed_dict = {x: train_data_batch}
  train_res = sess.run([train_op, x_recon, recon_error, loss, perplexity], feed_dict=train_feed_dict)
  
  train_res_recon_error.append(train_res[2])
  train_res_loss.append(train_res[3])
  train_res_perplexity.append(train_res[4])

  if i == 0 or (i+1) % 100 == 0:
    print('%d iterations | loss:  %.3f | recon_error: %.3f | perplexity: %.3f ' % (i+1, np.mean(train_res_loss[-100:]), np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))
    x_recon_img = train_res[1]

    test_data_batch = sess.run(test_dataset_batch_tf)
    test_feed_dict = {x: test_data_batch}
    [test_recon_img, test_recon_img_rand, test_recon_img_near] = sess.run([test_recon, test_recon_rand, test_recon_near], feed_dict=test_feed_dict)

    f = plt.figure(figsize=(16,24))
    ax = f.add_subplot(3,2,1)
    ax.imshow(convert_batch_to_image_grid(train_data_batch), interpolation='nearest', cmap ='gray')
    ax.set_title('training data originals')
    plt.axis('off')

    ax = f.add_subplot(3,2,2)
    ax.imshow(convert_batch_to_image_grid(x_recon_img), interpolation='nearest', cmap ='gray')
    ax.set_title('training data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(3,2,3)
    ax.imshow(convert_batch_to_image_grid(test_data_batch), interpolation='nearest', cmap ='gray')
    ax.set_title('test data originals')
    plt.axis('off')

    ax = f.add_subplot(3,2,4)
    ax.imshow(convert_batch_to_image_grid(test_recon_img), interpolation='nearest', cmap ='gray')
    ax.set_title('test data reconstructions')
    plt.axis('off')

    ax = f.add_subplot(3,2,5)
    ax.imshow(convert_batch_to_image_grid(test_recon_img_near), interpolation='nearest', cmap ='gray')
    ax.set_title('test data recon by near encoder')
    plt.axis('off')

    ax = f.add_subplot(3,2,6)
    ax.imshow(convert_batch_to_image_grid(test_recon_img_rand), interpolation='nearest', cmap ='gray')
    ax.set_title('data recon by random encoder')
    plt.axis('off')

    plt.savefig(save_res_dir+'/reconstructions_iter_'+str(i+1)+'.png')
    plt.close()


f = plt.figure(figsize=(24,8))
ax = f.add_subplot(1,3,1)
ax.plot(train_res_recon_error)
ax.set_yscale('log')
ax.set_title('train recon_error.')

ax = f.add_subplot(1,3,2)
ax.plot(train_res_loss)
ax.set_yscale('log')
ax.set_title('loss.')

ax = f.add_subplot(1,3,3)
ax.plot(train_res_perplexity)
ax.set_title('Average codebook usage (perplexity).')

plt.savefig(save_res_dir+"/loss.png")
plt.close()

#--- Train pixelcnn
train_loss_pixelcnn = []
for i in range(num_training_updates_pixelcnn):
    train_data_batch = sess.run(train_dataset_batch_tf)
    train_databatch_pixelcnn_np = sess.run(train_databatch_pixelcnn, feed_dict={x: train_data_batch})
    train_feed_dict = {x: train_data_batch, data_pixelcnn: train_databatch_pixelcnn_np}
    train_res_pixelcnn = sess.run([loss_pixelcnn, optimizer_pixelcnn], feed_dict=train_feed_dict)

    train_loss_pixelcnn.append(train_res_pixelcnn[0])

    if i == 0 or (i+1) % 100 == 0:
        print('%d iter_pixelcnn | loss:  %.3f ' % (i+1, train_res_pixelcnn[0]))
        n_row = 10
        n_col = 10
        samples = np.zeros(shape=(n_row*n_col, code_size, code_size), dtype=np.int32)
        for j in range(code_size):
            for k in range(code_size):
                data_dict = {data_pixelcnn: samples}
                next_sample = sess.run(sampled_pixelcnn_train, feed_dict=data_dict)
                samples[:, j, k] = next_sample[:, j, k]
        samples.astype(np.int32)
        
        feed_dict = {x: train_data_batch, sampled_code_pixelcnn: samples}
        x_recon_pixelcnn_res = sess.run(x_recon_pixelcnn, feed_dict=feed_dict)

        f = plt.figure(figsize=(6,16))
        ax = f.add_subplot(2,1,1)
        ax.imshow(convert_batch_to_image_grid(train_data_batch), interpolation='nearest', cmap ='gray')
        ax.set_title('training data originals')
        plt.axis('off')

        ax = f.add_subplot(2,1,2)
        ax.imshow(convert_batch_to_image_grid(x_recon_pixelcnn_res), interpolation='nearest', cmap ='gray')
        ax.set_title('samples')
        plt.axis('off')

        plt.savefig(save_res_dir+'/sampled_iter_'+str(i+1)+'.png')
        plt.close()

