import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell

from utils import *

class LSTMCellWithAtt(RNNCell):
    def __init__(self, H_q, hidden_size):
        self.H_q = H_q

        shape = tf.shape(H_q)
        self.batch_size = shape[0]
        self.Q = shape[1]

        self.hidden_size = hidden_size
        self.cell = BasicLSTMCell(self.hidden_size)

    @property
    def state_size(self):
        # Assume we have state_is_tuple, so we can interface
        # with the LSTM cell underneath
        return (self.hidden_size, self.hidden_size)

    @property
    def output_size(self):
        return self.hidden_size

    def __call__(self, inputs, state, scope=None):
        _, h_p = state # Ignore cell state, keep h

        with tf.variable_scope(type(self).__name__):
            with tf.variable_scope("weights", initializer = tf.contrib.layers.xavier_initializer()):
                W_q = tf.get_variable("W_q", shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
                W_p = tf.get_variable("W_p", shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
                W_r = tf.get_variable("W_r", shape=[self.hidden_size, self.hidden_size], dtype=tf.float32)
                W = tf.get_variable("W", shape=[self.hidden_size, 1], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("biases", initializer = tf.constant_initializer(0.0)):
                b_p = tf.get_variable("b_p", shape=[self.hidden_size], dtype=tf.float32)
                b = tf.get_variable("b", shape=[1], dtype=tf.float32)

            term1 = batch_matmul(self.H_q, W_q)
            assert_rank("term1", term1, expected_rank=3)

            # Second term
            term2 = tf.matmul(inputs, W_p)
            term2 = tf.reshape(term2, [self.batch_size, 1, self.hidden_size])

            term3 = tf.matmul(h_p, W_r)
            term3 = tf.reshape(term3, [self.batch_size, 1, self.hidden_size])

            G = tf.nn.tanh(term1 + term2 + term3 + b_p)
            G = tf.reshape(G, [self.batch_size * self.Q, self.hidden_size])
            assert_dim("G", G, dim=1, expected_value=self.hidden_size)

            # Reshape before multiplication
            alpha = tf.nn.softmax(tf.matmul(G, W) + b)
            alpha = tf.reshape(alpha, [self.batch_size, self.Q, 1])
            # H_q = tf.reshape(H_q, [self.batch_size, self.Q, self.hidden_size])

            attended = tf.matmul(tf.transpose(alpha, [0, 2, 1]), self.H_q)
            # import pdb; pdb.set_trace()
            inputs = tf.reshape(inputs, [self.batch_size, 1, self.hidden_size])
            z_i = tf.concat([inputs, attended], axis=2)
            z_i = tf.reshape(z_i, [self.batch_size, 2*self.hidden_size])
            # Unroll this shit
            output, states = self.cell(z_i, state)
            return output, (states.c, states.h)

class AnsPtrCell(RNNCell):
    def __init__(self, Hr, hidden_size):
        self._Hr = Hr

        shape = tf.shape(Hr)
        self._batch_size = shape[0]
        self._P = shape[1]

        self._hidden_size = hidden_size
        self._cell = BasicLSTMCell(hidden_size)

    @property
    def state_size(self):
        """State tuple"""
        return (self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        """Returns a probability distribution over the paragraph tokens."""
        return self._P

    def __call__(self, _, state):
        _, h_a = state

        with tf.variable_scope(type(self).__name__):
            with tf.variable_scope("weights", initializer = tf.contrib.layers.xavier_initializer()):
                V = tf.get_variable("V", shape=[2*self._hidden_size, self._hidden_size], dtype=tf.float32)
                W_a = tf.get_variable("W_a", shape=[self._hidden_size, self._hidden_size], dtype=tf.float32)
                v = tf.get_variable("v", shape=[self._hidden_size, 1], dtype=tf.float32)

            with tf.variable_scope("biases", initializer = tf.constant_initializer(0.0)):
                b_a = tf.get_variable("b_a", shape=[1, self._hidden_size], dtype=tf.float32)
                c = tf.get_variable("c", shape=[1], dtype=tf.float32)

            x = batch_matmul(self._Hr, V)   # H_r is [batch, P, 2*hidden_size],   V is [2*hidden_size, hidden_size]. Output is [batch_size, P, hidden_size]
            assert_rank("x", x, expected_rank=3)
            # assert_dim("x", x, dim=2, expected_value=self._hidden_size)


            y = tf.matmul(h_a, W_a)  # h should be [batch_size, hidden_size],   W_a is [hidden_size, hidden_size] ==> output is [batch_size, 1, hidden_size]
            y = tf.reshape(y, [self._batch_size, 1, self._hidden_size])
            assert_rank("y", y, expected_rank=3)

            z = b_a                # b_a is [hidden_size], should broadcast
            q = x + y + z

            F = tf.nn.tanh(q)       # tanh across, get [batch_size, P, hidden_size]
            assert_rank("F", F, expected_rank=3)

            z = batch_matmul(F, v) + c
            assert_rank("z", z, expected_rank=3)

            z = tf.Print(z, [z, tf.shape(z)])
            B = tf.nn.softmax(z, dim=1)        # F is [batch_size, P, hidden_size], v is [hidden_size, 1], output is [batch_size, P, 1]
            B = tf.Print(B, [B])
            assert_rank("B", B, expected_rank=3)

            B = tf.reshape(B, [self._batch_size, self._P, 1])

            prod = tf.matmul(tf.transpose(B, [0, 2, 1]), self._Hr)       # H_r is [batch_size, P, 2*hidden_size],  B is [batch_size, P, 1]. Just make sure B was reshaped before multiply
            prod = tf.reshape(prod, [self._batch_size, 2*self._hidden_size])

            _, state = self._cell(prod, state)
            return B, (state.c, state.h)    # We want to propagate the B(eta)'s across
