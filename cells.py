import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell

from utils import batch_matmul

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
        scope = scope or type(self).__name__

        h_p = state[1] # Ignore cell state, keep h
        print("h_p size:", h_p.get_shape())

        # Include the state input
        # import pdb; pdb.set_trace()

        # Split the state if it's a tuple

        with tf.variable_scope(scope):
            W_q = tf.get_variable("W_q", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W_p = tf.get_variable("W_p", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W_r = tf.get_variable("W_r", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W = tf.get_variable("W", [self.hidden_size, 1], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable("b_p", [self.hidden_size], tf.float32, initializer = tf.constant_initializer(0.0))
            b = tf.get_variable("b", [1], tf.float32, initializer = tf.constant_initializer(0.0))

            # Use H_q times W_q
            # Reshape this
            H_q = tf.reshape(self.H_q, [self.batch_size * self.Q, self.hidden_size])
            term1 = tf.matmul(H_q, W_q)
            term1 = tf.reshape(term1, [self.batch_size, self.Q, self.hidden_size])

            # Second term
            term2 = tf.matmul(inputs, W_p)
            term2 = tf.reshape(term2, [self.batch_size, 1, self.hidden_size])

            term3 = tf.matmul(h_p, W_r)
            term3 = tf.reshape(term3, [self.batch_size, 1, self.hidden_size])

            G = tf.nn.tanh(term1 + term2 + term3 + b_p)
            G = tf.reshape(G, [self.batch_size * self.Q, self.hidden_size])

            assert G.get_shape()[1] == self.hidden_size, "G needs hidden_size in last dim, was {}".format(G.get_shape())

            # H_q   = batch x Q x L
            # alpha = batch x Q x 1
            # We need to multiply alpha^T H_q
            # This gives us batch x 1 x L
            # z_i is batch x 1 x 2L

            # Reshape before multiplication
            alpha = tf.nn.softmax(tf.matmul(G, W) + b)
            alpha = tf.reshape(alpha, [self.batch_size, self.Q, 1])
            H_q = tf.reshape(H_q, [self.batch_size, self.Q, self.hidden_size])

            attended = tf.matmul(tf.transpose(alpha, [0, 2, 1]), H_q)
            # import pdb; pdb.set_trace()
            inputs = tf.reshape(inputs, [self.batch_size, 1, self.hidden_size])
            z_i = tf.concat([inputs, attended], axis=2)
            z_i = tf.reshape(z_i, [self.batch_size, 2*self.hidden_size])
            # import pdb; pdb.set_trace()
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
        return (self._hidden_size, self._hidden_size)

    @property
    def output_size(self):
        raise NotImplementedError("fill this in if we need to")

    def __call__(self, inputs, state, scope=None, reuse=False):
        scope = scope or type(self).__name__

        h = state[1]
        # print("h size:", h.get_shape())
        # print("state size:", state.get_shape())

        # import pdb; pdb.set_trace()

        with tf.variable_scope(scope, reuse=reuse):
            # import pdb; pdb.set_trace()
            V = tf.get_variable("V", [2*self._hidden_size, self._hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W_a = tf.get_variable("W_a", [self._hidden_size, self._hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b_a = tf.get_variable("b_a", [self._hidden_size], tf.float32, initializer = tf.constant_initializer(0.0))
            v = tf.get_variable("v", [self._hidden_size, 1], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            c = tf.get_variable("c", [1], tf.float32, initializer = tf.constant_initializer(0.0))

            F = tf.nn.tanh(batch_matmul(self._Hr, V) + tf.matmul(h, W_a) + b_a)
            B = tf.nn.softmax(batch_matmul(F, v) + c)
            B = tf.reshape(B, [self._batch_size, 1, self._P])

            prod = tf.matmul(B, self._Hr)
            print("H_r size:", self._Hr.get_shape())    # H_r is size [batchsize, P, 2*L]
            prod = tf.reshape(prod, [self._batch_size, 2*self._hidden_size])
            # import pdb; pdb.set_trace()

            _, state = self._cell(prod, state)
            return B, (state.c, state.h)

            # Output here is the next B
