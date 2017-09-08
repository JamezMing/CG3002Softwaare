import tensorflow as tf
import dataImport

dataSet = dataImport.dataFile()



learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 6 # MNIST data input (img shape: 28*28)
timesteps = 100 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])