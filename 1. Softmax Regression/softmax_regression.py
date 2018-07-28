# importing MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("- MNIST dataset imported\n")

# output the format of train, test, and validation data
print("- Trainning data and label dimension:")
print(mnist.train.images.shape, mnist.train.labels.shape)
print("- Testing data and label dimension:")
print(mnist.test.images.shape, mnist.test.labels.shape)
print("- Validation data and label dimension:")
print(mnist.validation.images.shape, mnist.validation.labels.shape)
print

# create tensorflow session
import tensorflow as tf 
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32, [None, 784])

# initialize variables to zeros for softmax regression
W = tf.Variable(tf.zeros([784, 10])) # 784 dim with 10 classes
b = tf.Variable(tf.zeros([10]))

# use built-in softmax regression in tf.nn: y = softmax(Wx + b)
y = tf.nn.softmax(tf.matmul(x, W) + b)

# define loss function with cross entropy input with label
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# train with gradient descent optimizer with 1000 iterations
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
tf.global_variables_initializer().run()
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

# evaluate the accuracy by comparing predicted labels with true labels
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("- Accuracy:"),
print(accuracy.eval({x: mnist.test.images, y_: mnist.test.labels}))