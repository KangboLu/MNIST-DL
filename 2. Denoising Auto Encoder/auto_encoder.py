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
    all_weights['w2'] = tf.Variable(tf.zeros([self.hidden, self.n_input], dtype = tf.float32)) # set reconstruction weight to 0
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