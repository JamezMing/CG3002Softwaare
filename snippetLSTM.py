import tensorflow as tf
import numpy as np
import dataImport
from tensorflow.contrib import rnn

learning_rate = 0.001
training_steps = 100000
batch_size = 100
display_step = 200

# Network Parameters
num_input = 3 # Each time stamp includes a 3-d data
timesteps = 30 # window size of the data
num_hidden = 256 # hidden layer num of features
num_classes = 6 # MNIST total classes (0-9 digits)

dataSet = dataImport.dataFile()
dataSet.init_dataset()

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


def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred_logits = RNN(X, weights, biases)
pred = tf.nn.softmax(pred_logits)
loss_op = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = Y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = dataSet.sliding_window_batches(step_size=10)
        array_x = np.array(batch_x)
        array_y = np.array(batch_y)
        # Reshape data to get 28 seq of 28 elements
        array_x = array_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: array_x, Y: array_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: array_x,
                                                                 Y: array_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    save_path = saver.save(sess, "./model.ckpt")
    print("Model saved in file: %s" % save_path)

print("Optimization Finished!")