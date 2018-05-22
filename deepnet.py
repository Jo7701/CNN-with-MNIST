import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("/tmp/data", one_hot = True)


n_classes = 10
batch_size = 100
width = 28
height = 28

x = tf.placeholder('float', [None, height, width, 1], name='x')
y = tf.placeholder('float', name = 'y')

def convolutional_layer(input, filter_shape, pool_shape, num_input, num_output):
	filter = [filter_shape[0], filter_shape[1], num_input, num_output]
	weights = tf.Variable(tf.random_normal(filter))
	bias = tf.Variable(tf.random_normal([num_output]))
	dotted_input = tf.nn.relu(tf.nn.conv2d(input, weights, strides = [1,1,1,1], padding = 'SAME') + bias)
	pool = [1,pool_shape[0], pool_shape[1], 1]
	pool_strides = [1,1,1,1]
	max_pool = tf.nn.max_pool(dotted_input, pool, pool_strides, padding = 'SAME')
	return max_pool

def convolutional_neural_network(input, num_layers):
	layer = convolutional_layer(input, (3,3), (2,2), 1, 8)
	for x in range(1, num_layers):
		layer = convolutional_layer(layer, (3,3), (2,2), int(layer.shape[-1]), int(layer.shape[-1])*2)
	flattened = tf.reshape(layer, (1, -1))
	weights = tf.Variable(tf.random_normal([int(flattened.shape[1]), n_classes]))
	bias = tf.Variable(tf.random_normal([n_classes]))
	output = tf.add(tf.matmul(flattened, weights), bias, name = 'output')
	return output


def train(x):
	prediction = convolutional_neural_network(x, 1)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	n_epochs = 5
	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(mnist.train.num_examples / batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape([batch_size, height, width, 1])
				_, c  = sess.run([optimizer, cost], feed_dict={x: epoch_x, y:epoch_y})
				epoch_loss += c
			print 'Epoch ', epoch + 1, ' Loss: ', epoch_loss


		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print 'Accuracy: ', accuracy.eval({x:mnist.test.images.reshape(-1, height, width, 1), y:mnist.test.labels})

train(x)
