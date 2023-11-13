import tensorflow as tf

def vector_quantizer(inputs, embedding_dim, num_embeddings, commitment_cost, random_gen=False,
	only_lookup=False, inputs_indices=None):
	'''
	Note: 
		shape_of_inputs=[batch_size, ?, ?, embedding_dim]
	'''
	# Assert last dimension of inputs is same as embedding_dim
	
	
	# assert_dim = tf.debugging.assert_equal(tf.shape(inputs)[-1], embedding_dim)
	# with tf.control_dependencies([assert_dim]):
	# 	flat_inputs = tf.reshape(inputs, [-1, embedding_dim])
	flat_inputs = tf.reshape(inputs, [-1, embedding_dim])


	with tf.compat.v1.variable_scope('vq', reuse=tf.compat.v1.AUTO_REUSE):
		emb_vq = tf.compat.v1.get_variable(name='embedding_vq', shape=[embedding_dim, num_embeddings], initializer=tf.compat.v1.uniform_unit_scaling_initializer())

	if (only_lookup == False):
		distances = tf.reduce_sum(flat_inputs**2, 1, keepdims=True) - 2*tf.matmul(flat_inputs, emb_vq) + tf.reduce_sum(emb_vq**2, 0, keepdims=True)
		encoding_indices = tf.argmax(-distances, 1)
		encodings = tf.one_hot(encoding_indices, num_embeddings)
		encoding_indices = tf.reshape(encoding_indices, tf.shape(inputs)[:-1]) # shape=[batch_size, ?, ?]
	else:
		inputs_indices = tf.cast(inputs_indices, tf.int32)
		encoding_indices = inputs_indices
		encodings = tf.one_hot(tf.reshape(encoding_indices, [-1,]), num_embeddings)

	quantized = tf.nn.embedding_lookup(tf.transpose(emb_vq), encoding_indices) 
	# Important Note: 
	# 	quantized is differentiable w.r.t. tf.transpose(emb_vq), 
	#	but is not differentiable w.r.t. encoding_indices.

	inp_latent_loss = tf.reduce_mean((tf.stop_gradient(quantized) - inputs)**2)
	emb_latent_loss = tf.reduce_mean((quantized - tf.stop_gradient(inputs))**2)
	loss = emb_latent_loss + commitment_cost*inp_latent_loss # used to optimize emb_vq only!

	quantized = inputs + tf.stop_gradient(quantized-inputs) 
	# Important Note: 
	# 	This step is used to copy the gradient from inputs to quantized.

	avg_probs = tf.reduce_mean(encodings, 0)
	perplexity = tf.exp(- tf.reduce_sum(avg_probs * tf.math.log(avg_probs+1e-10)))
	# The perplexity is the exponentiation of the entropy, 
	# indicating how many codes are 'active' on average.
	# We hope the perplexity is larger.

	if (random_gen == False):
		return {'quantized': quantized,
				'loss': loss,
				'perplexity': perplexity,
				'encodings': encodings,
				'encoding_indices': encoding_indices}
	else:
		rand_encoding_indices = tf.random.uniform(tf.shape(encoding_indices), minval=0, maxval=1)
		rand_encoding_indices = tf.floor(rand_encoding_indices * num_embeddings)
		rand_encoding_indices = tf.clip_by_value(rand_encoding_indices, 0, num_embeddings-1)
		rand_encoding_indices = tf.cast(rand_encoding_indices, tf.int32)

		rand_quantized = tf.nn.embedding_lookup(tf.transpose(emb_vq), rand_encoding_indices) 

		near_encoding_indices = tf.cast(encoding_indices, tf.float32) + tf.random.uniform(tf.shape(encoding_indices), minval=-1, maxval=1)
		near_encoding_indices = tf.clip_by_value(near_encoding_indices, 0, num_embeddings-1)
		near_encoding_indices = tf.round(near_encoding_indices)
		near_encoding_indices = tf.cast(near_encoding_indices, tf.int32)
		
		near_quantized = tf.nn.embedding_lookup(tf.transpose(emb_vq), near_encoding_indices)

		return {'quantized': quantized,
				'loss': loss,
				'perplexity': perplexity,
				'encodings': encodings,
				'encoding_indices': encoding_indices,
				'rand_quantized': rand_quantized,
				'near_quantized': near_quantized}


# import tensorflow as tf
# import numpy as np

# x = tf.ones([2, 3, 3, 32])
# outputs = vector_quantizer(inputs=x, embedding_dim=32, num_embeddings=9, commitment_cost=0.5, random_gen=True)

# sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# out = sess.run(outputs)
# print(out)


