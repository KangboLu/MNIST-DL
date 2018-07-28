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