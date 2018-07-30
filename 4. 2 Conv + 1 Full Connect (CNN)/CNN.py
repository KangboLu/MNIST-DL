 # import MNIST dataset and tensorflow
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create a interactive session
sess = tf.InteractiveSession()

# helper function to initialize weight
def weight_initializer(shape):
  return tf.Variable(tf.truncated_normal(shape, stddev=0.1))

# heper function to initialize bias
def bias_initializer(shape):
  return tf.Variable(tf.constant(0.1, shape=shape)) # avoid dead neurons

# helper function to create 2 dimensional convolution layer
def conv2d(x, w):
  return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

# heper function to create max pooling layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# create placeholders
x = tf.placeholder(tf.float32, [None, 784]) # features
x_image = tf.reshape(x, [-1,28,28,1]) # reshape 1D input to 2D structure

# define first convolution layer
w_conv1 = weight_initializer([5,5,1,32]) # kernel size 5x5, 1 channel, 32 different kernel
b_conv1 = bias_initializer([32]) # 32 bias for each kernel
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1) # relu activation with input x_image
h_pool1 = max_pool_2x2(h_conv1) # extract max with max pooling

# define second convolution layer
w_conv2 = weight_initializer([5,5,32,64]) # kernel size 5x5, 32 channel, 64 different kernel
b_conv2 = bias_initializer([64]) # 64 bias for each kernel
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2) # relu activation with input h_pool1
h_pool2 = max_pool_2x2(h_conv2) # extract max with max pooling

# define full connection layer
h_pool2_1d = tf.reshape(h_pool2, [-1, 7*7*64]) # reshape 2nd conv layer output to 1d
w_fc1 = weight_initializer([7*7*64, 1024])
b_fc1 = bias_initializer([1024])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_1d, w_fc1) + b_fc1)

# define dropout layer to prevent overfitting
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# define softmax layer with dropout layer's output as input
w_fc2 = weight_initializer([1024, 10])
b_fc2 = bias_initializer([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

# define loss function and use Adam optimizer
y_ = tf.placeholder(tf.float32, [None, 10]) # labels
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv),
  reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# define how accuracy calculation
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize global variables
tf.global_variables_initializer().run()

# training steps
for i in range(20000):
  batch = mnist.train.next_batch(50)
  # for every 100 iteration, evaluate the accuracy
  if i % 100 == 0:
    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
    print("Step %d, trainning accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

# finish training, test the model
print("Test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, 
  y_: mnist.test.labels, keep_prob: 1.0}))