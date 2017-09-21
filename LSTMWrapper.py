import tensorflow as tf
import numpy as np
import dataImport
from tensorflow.contrib import rnn

class LSTM_Model(object):
    def __init__(self, ckpt_path = ''):
        self.learning_rate = 0.001
        self.training_steps = 100000
        self.batch_size = 40
        self.display_step = 200
        self.timesteps = 30   # window size of the data
        self.train_set = dataImport.dataFile(batch_size=self.batch_size, window_size=self.timesteps, step_size= 5)
        self.train_set.init_dataset()

        # Network Parameters
        self.num_input = 3  # Each time stamp includes a 3-d data
        self.num_hidden = 128  # hidden layer num of features
        self.num_classes = 6  # MNIST total classes (0-9 digits)
        self.model_name = 'lstm_default'
        self.ckpt_path = ckpt_path

        def __init_graph__():
            tf.reset_default_graph()
            self.input = tf.placeholder(dtype="float", shape=[None, self.timesteps, self.num_input])
            self.target = tf.placeholder(dtype="float", shape=[None, self.num_classes])
            self.weights = {
                'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
            }
            self.biases = {
                'out': tf.Variable(tf.random_normal([self.num_classes]))
            }
            lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0)
            with tf.variable_scope('lstm') as scope:
                inputs = tf.unstack(self.input, self.timesteps, 1)
                self.outputs, self.states = rnn.static_rnn(lstm_cell, inputs, dtype=tf.float32)
            pred_logits = tf.matmul(self.outputs[-1], self.weights['out']) + self.biases['out']
            pred = tf.nn.softmax(pred_logits)
            self.loss_op = tf.reduce_mean((tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=self.target)))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            self.train_op = optimizer.minimize(self.loss_op)
        __init_graph__()

    def train_batch(self, sess, dataset):
        batch_x, batch_y = dataset.sliding_window_batches()
        array_x = np.array(batch_x)
        array_y = np.array(batch_y)
        # Reshape data to get 28 seq of 28 elements
        array_x = array_x.reshape((self.batch_size, self.timesteps, self.num_input))
        feed_dict = {self.input: array_x, self.target: array_y}
        _, loss = sess.run([self.train_op, self.loss_op], feed_dict)
        return loss

    def restore_last_session(self):
        saver = tf.train.Saver()
        # create a session
        sess = tf.Session()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        # return to user
        return sess

    def predict(self, sess, test_data):
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
        feed_data = test_data
        out = sess.run(self.outputs, feed_dict= {self.input:feed_data})
        pred_logits = tf.matmul(out[-1], self.weights['out']) + self.biases['out']
        pred = tf.nn.softmax(pred_logits)
        return sess.run(pred)



    def train(self, sess=None):
        # we need to save the model periodically
        saver = tf.train.Saver()
        # if no session is given
        if not sess:
            # create a session
            sess = tf.Session()
            # init all variables
            sess.run(tf.global_variables_initializer())
        # run M epochs
        for i in range(self.training_steps):
            try:
                self.train_batch(sess = sess, dataset = self.train_set)
                if i and i % (5000) == 0:  # TODO : make this tunable by the user
                    # save model to disk
                    saver.save(sess, "./" + self.model_name + '.ckpt', global_step=i)
                    # evaluate to get validation loss
                    # print stats
                    print('\nModel saved to disk at iteration #{}'.format(i))
            except KeyboardInterrupt:  # this will most definitely happen, so handle it
                print('Interrupted by user at iteration {}'.format(i))
                self.session = sess
                return sess
        saver.save(sess, "./" + self.model_name + '.ckpt', global_step=self.training_steps)


model = LSTM_Model()
#model.train()
sess = model.restore_last_session()
data = dataImport.dataFile(batch_size=40, window_size=30, step_size=5)
data.init_dataset()
tdata, tlabel = data.get_test_set()
print(data.__data_stats__(tlabel))
pred = model.predict(sess, tdata)
conf_mat = np.zeros((6,6))
print (pred.shape)
acc_count = 0
res = np.zeros(pred.shape)

for i in range(pred.shape[0]):
    pred_in = np.argmax(pred[i])
    actual_in = np.argmax(tlabel[i])
    conf_mat[actual_in][pred_in] = conf_mat[actual_in][pred_in] + 1

TP = np.zeros((6,1))
FP = np.zeros((6,1))
FN = np.zeros((6,1))
NumEle = np.zeros((6,1))

for i in range(0,6):
    TP[i] = conf_mat[i][i] + TP[i]
    NumEle[i] = sum(conf_mat[i])

for i in range(0,6):
    for j in range(0,6):
        if i!=j:
            FP[i] = conf_mat[i][j] + FP[i]
            FN[j] = conf_mat[i][j] + FN[j]

pre = np.divide(TP*1.0,(TP+FP))
rec = np.divide(TP, (FN + TP))
F1 = 2*np.divide((np.multiply(pre, rec)), (pre + rec))
acc = np.divide(TP, NumEle)

print("F1 score is: " + str(np.average(F1)))
print("Recall is: " + str(np.average(rec)))
print("Precision is: " + str(np.average(pre)))
print (conf_mat)








