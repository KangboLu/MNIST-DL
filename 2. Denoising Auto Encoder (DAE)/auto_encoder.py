# include necessary packages and dataset
import numpy as np 
import sklearn.preprocessing as prep
import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

# xavier initialization to adjust the weight to keep signals within decent range
def xavier_init(fan_in, fan_out, constant = 1):
	low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
	high = constant * np.sqrt(6.0 / (fan_in + fan_out))
	return tf.random_uniform((fan_in, fan_out),
            							 minval = low, maxval = high,
            							 dtype = tf.float32)

# class definition of additive aussian noise auto encoder
class AdditiveGaussianNoiseAutoEncoder(object):
  # constructor to set up neural network's specs
  def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, 
               optimizer = tf.train.AdamOptimizer(), scale = 0.1):
    self.n_input = n_input
    self.n_hidden = n_hidden
    self.transfer = transfer_function
    self.scale = tf.placeholder(tf.float32)
    self.training_scale = scale
    network_weights = self.initialize_weights()
    self.weights = network_weights # assign network weights to self.weight

    # network model
    self.x = tf.placeholder(tf.float32, [None, self.n_input])
    self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
                  self.weights['w1']),
                  self.weights['b1']))
    self.reconstruction = tf.add(tf.matmul(self.hidden,
                            self.weights['w2']), self.weights['b2'])

    # define cost with square error
    self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
    self.optimizer = optimizer.minimize(self.cost)

    # initialization of the model specs with session
    init = tf.global_variables_initializer()
    self.sess = tf.Session()
    self.sess.run(init)

  # initialization of network weights for the above constructor
  def initialize_weights(self):
    # create a dictionary to store the weights for hidden and reconstruction layer nodes
    all_weights = dict()
    all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden)) # set weight with xavier init
    all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32)) # set bias to 0
    all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32)) # set reconstruction weight to 0
    all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32)) # set bias to 0
    return all_weights

  # calculate cost with defined cost and optimizer in the constructor
  def partial_fit(self, X):
    cost, opt = self.sess.run((self.cost, self.optimizer), 
                  feed_dict = {self.x: X, self.scale: self.training_scale})
    return cost

  # calculate total cost of the network
  def calc_total_cost(self, X):
    return self.sess.run(self.cost, feed_dict = {self.x: X, self.scale: self.training_scale})

  # transform function to return hidden layer output
  def transform(self, X):
    return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scale})

  # generate function to pass hidden later output to reconstruction layer
  def generate(self, hidden = None):
    if hidden is None:
      hidden = np.random.normal(size = self.weights['b1'])
    return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

  # reconstruct the data by extracting high level features
  def reconstruct(self, X):
    return self.sess.run(self.reconstruction, feed_dict = {self.x: X, self.scale: self.training_scale})

  # get hidden level weights w1
  def get_weights(self):
    return self.sess.run(self.weights['w1'])

  # get bias from hidden layer
  def get_biases(self):
    return self.sess.run(self.weights['b1'])

# ===============================================================
# testing Additive Gaussian Noise Auto Encoder with MNIST dataset
# ===============================================================

# read MNIST dataset with one hot encoding
mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

# function to obtain random block of data
def get_random_block(data, batch_size):
  start_index = np.random.randint(0, len(data) - batch_size)
  return data[start_index:(start_index + batch_size)]

# standardize traning and test data
def standard_scale(X_train, X_test):
  preprocessor = prep.StandardScaler().fit(X_train)
  X_train = preprocessor.transform(X_train)
  X_test = preprocessor.transform(X_test)
  return X_train, X_test
X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

# set up parameters for training
n_samples = int(mnist.train.num_examples)
training_epochs = 20
batch_size = 128
display_step = 1 # display cost for each epoch

# create AdditiveGaussianNoiseAutoEncoder instance
auto_encoder = AdditiveGaussianNoiseAutoEncoder(n_input = 784,
                 n_hidden = 200,
                 transfer_function = tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(learning_rate = 0.001),
                 scale = 0.01)

# start training
for epoch in range(training_epochs):
  avg_cost = 0
  total_batch = int(n_samples / batch_size)

  # iterate all the batches
  for i in range(total_batch):
    batch_xs = get_random_block(X_train, batch_size)

    # obatin cost by parsing batch to fit training
    cost = auto_encoder.partial_fit(batch_xs)
    avg_cost += cost / n_samples * batch_size # increment avg cost

  # print each epoch's cost
  if epoch % display_step == 0:
    print("epoch: ", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# output total cost by parsing test data
print("Total cost: " + str(auto_encoder.calc_total_cost(X_test)))