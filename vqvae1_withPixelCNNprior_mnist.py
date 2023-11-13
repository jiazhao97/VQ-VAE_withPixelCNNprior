import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os

import networks_vqvae1 as vqvae_nets
import vector_quantizer as vq
import networks_pixelcnn as pixelcnn_nets

#--------------------------------------------------------------------------
# Set hyper-parameters
save_res_dir = "0820_res_vqvae1_PixelCNN_mnist_K8_D16_gradclip_n20000_lr1e_3_batch100"
if not os.path.exists(save_res_dir):
    os.makedirs(save_res_dir)

# for vqvae1
num_training_updates = 20000
image_size = 28
num_channel = 1
batch_size = 100

num_hiddens = 64
num_residual_hiddens = 16
num_residual_layers = 2

embedding_dim = 16 # D
num_embeddings = 8 # K
commitment_cost = 0.25

learning_rate = 3e-4

# for PixelCNN
num_training_updates_pixelcnn = 30000

num_layers_pixelcnn = 12
fmaps_pixelcnn = 32
code_size = 7

learning_rate_pixelcnn = 1e-3
grad_clip_pixelcnn = 1.0

#--------------------------------------------------------------------------
# Placeholder
x = tf.keras.Input(shape=(image_size, image_size, num_channel))
data_pixelcnn = tf.keras.Input(shape=(None, 7, 7)) # Train
sampled_code_pixelcnn = tf.keras.Input(shape=(None, 7, 7)) # Plot
#--------------------------------------------------------------------------
# Data

# Tools to convert images to floating point with the range [-0.5, 0.5]
def cast_and_normalise_images(images):
    images = tf.cast(images, tf.float32) - 0.5
    return images

# Tools to reconstruct data
def convert_batch_to_image_grid(images, image_size=image_size, num_channel=num_channel):
    reshaped = (images.reshape(10, 10, image_size, image_size)
                .transpose(0, 2, 1, 3)
                .reshape(10 * image_size, 10 * image_size))
    return reshaped + 0.5

from tensorflow.keras.datasets import mnist
(mnist_train_images, _), (mnist_test_images, _) = mnist.load_data()

mnist_train_images = mnist_train_images.reshape([-1, 28, 28, 1]) # shape=[55000, 28, 28, 1]
mnist_test_images = mnist_test_images.reshape([-1, 28, 28, 1])   # shape=[10000, 28, 28, 1]

data_variance = np.var(mnist_train_images)

train_dataset = (tf.data.Dataset.from_tensor_slices(mnist_train_images)
                    .map(cast_and_normalise_images)
                    .shuffle(10000)
                    .repeat()
                    .batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(mnist_test_images)
                .map(cast_and_normalise_images)
                .repeat()
                .batch(batch_size))
train_dataset_batch_tf = next(iter(train_dataset))
test_dataset_batch_tf  = next(iter(test_dataset))
print("Data loading is finished...")





#--------------------------------------------------------------------------
# Training process

#--- Train process - vqvae1
#--- Train process - vqvae1
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
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

#--- Train process - PixelCNN for priors
train_databatch_pixelcnn = vq_output["encoding_indices"]
pixelcnn_output = pixelcnn_nets.pixelcnn(data_pixelcnn, num_layers_pixelcnn, fmaps_pixelcnn, num_embeddings, code_size)
loss_pixelcnn = pixelcnn_output["loss_pixelcnn"]
sampled_pixelcnn_train = pixelcnn_output["sampled_pixelcnn"]
trainer_pixelcnn = tf.optimizers.RMSprop(learning_rate=learning_rate_pixelcnn)
gradients_pixelcnn = trainer_pixelcnn.get_gradients(loss_pixelcnn, tf.trainable_variables())
clipped_gradients_pixelcnn = [(tf.clip_by_value(grad, -grad_clip_pixelcnn, grad_clip_pixelcnn), var) for grad, var in gradients_pixelcnn]
optimizer_pixelcnn = trainer_pixelcnn.apply_gradients(clipped_gradients_pixelcnn)

#--- For plots
vq_output_pixelcnn = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost, only_lookup=True, inputs_indices=sampled_code_pixelcnn)
x_recon_pixelcnn = vqvae_nets.decoder(vq_output_pixelcnn["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
test_vq_output = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost, random_gen=True)
test_recon = vqvae_nets.decoder(test_vq_output["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
test_recon_rand = vqvae_nets.decoder(test_vq_output["rand_quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
test_recon_near = vqvae_nets.decoder(test_vq_output["near_quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)

#--- Train vqvae1
train_res_recon_error = []
train_res_perplexity = []
train_res_loss = []

for i in range(num_training_updates):
  train_data_batch = next(iter(train_dataset))
  with tf.GradientTape() as tape:
    z = vqvae_nets.encoder(x, num_hiddens, num_residual_layers, num_residual_hiddens)
    z = vqvae_nets.conv2d(z, fmaps=embedding_dim, kernel=1, strides=1)
    vq_output = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost)
    x_recon = vqvae_nets.decoder(vq_output["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
    recon_error = tf.reduce_mean((x_recon - x)**2) / data_variance  # Normalized MSE
    loss = recon_error + vq_output["loss"]
    perplexity = vq_output["perplexity"] 
  gradients = tape.gradient(loss, tf.trainable_variables())
  optimizer.apply_gradients(zip(gradients, tf.trainable_variables()))
  
  train_res_recon_error.append(recon_error)
  train_res_loss.append(loss)
  train_res_perplexity.append(perplexity)
  if i == 0 or (i+1) % 100 == 0:
    print('%d iterations | loss:  %.3f | recon_error: %.3f | perplexity: %.3f ' % (i+1, np.mean(train_res_loss[-100:]), np.mean(train_res_recon_error[-100:]), np.mean(train_res_perplexity[-100:])))
    x_recon_img = x_recon
    test_data_batch = next(iter(test_dataset))
    z = vqvae_nets.encoder(x, num_hiddens, num_residual_layers, num_residual_hiddens)
    z = vqvae_nets.conv2d(z, fmaps=embedding_dim, kernel=1, strides=1)
    test_vq_output = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost)
    test_recon = vqvae_nets.decoder(test_vq_output["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
    test_recon_img = test_recon

#--- Train pixelcnn
train_loss_pixelcnn = []
for i in range(num_training_updates_pixelcnn):
    train_data_batch = next(iter(train_dataset))
    train_databatch_pixelcnn_np = vq_output["encoding_indices"]
    with tf.GradientTape() as tape:
        pixelcnn_output = pixelcnn_nets.pixelcnn(data_pixelcnn, num_layers_pixelcnn, fmaps_pixelcnn, num_embeddings, code_size)
        loss_pixelcnn = pixelcnn_output["loss_pixelcnn"]
        sampled_pixelcnn_train = pixelcnn_output["sampled_pixelcnn"]
    gradients_pixelcnn = tape.gradient(loss_pixelcnn, tf.trainable_variables())
    clipped_gradients_pixelcnn = [(tf.clip_by_value(grad, -grad_clip_pixelcnn, grad_clip_pixelcnn), var) for grad, var in gradients_pixelcnn]
    optimizer_pixelcnn.apply_gradients(clipped_gradients_pixelcnn)
    
    train_loss_pixelcnn.append(loss_pixelcnn)
    if i == 0 or (i+1) % 100 == 0:
        print('%d iter_pixelcnn | loss:  %.3f ' % (i+1, loss_pixelcnn))
        n_row = 10
        n_col = 10
        samples = np.zeros(shape=(n_row*n_col, code_size, code_size), dtype=np.int32)
        for j in range(code_size):
            for k in range(code_size):
                next_sample = pixelcnn_output["sampled_pixelcnn"]
                samples[:, j, k] = next_sample[:, j, k]
        samples.astype(np.int32)
        
        vq_output_pixelcnn = vq.vector_quantizer(z, embedding_dim, num_embeddings, commitment_cost, only_lookup=True, inputs_indices=sampled_code_pixelcnn)
        x_recon_pixelcnn = vqvae_nets.decoder(vq_output_pixelcnn["quantized"], num_hiddens, num_residual_layers, num_residual_hiddens, image_size, num_channel)
        f = plt.figure(figsize=(6,16))
        ax = f.add_subplot(2,1,1)
        ax.imshow(convert_batch_to_image_grid(train_data_batch), interpolation='nearest', cmap ='gray')
        ax.set_title('training data originals')
        plt.axis('off')
        ax = f.add_subplot(2,1,2)
        ax.imshow(convert_batch_to_image_grid(x_recon_pixelcnn), interpolation='nearest', cmap ='gray')
        ax.set_title('samples')
        plt.axis('off')
        plt.savefig(save_res_dir+'/sampled_iter_'+str(i+1)+'.png')
        plt.close()
