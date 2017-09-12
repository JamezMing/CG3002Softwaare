import tensorflow as tf
import dataImport
from tensorflow.contrib import rnn

dataSet = dataImport.dataFile()
dataSet.init_dataset()



learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 3 # Each time stamp includes a 3-d data
timesteps = 100 # window size of the data
num_hidden = 128 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)

# tf Graph input
#X input dimension: batch_size, window_size, imput_dim
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}
lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias = 1.0)
outputs, states = rnn.static_rnn(lstm_cell, X, dtype=tf.float32)
pred_logits = tf.matmul(outputs[-1], weights['out'] + biases['out'])
pred = tf.nn.softmax(pred_logits)