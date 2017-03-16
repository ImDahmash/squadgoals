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
            term2 = tf.expand_dims(term2, axis=1)

            term3 = tf.matmul(h_p, W_r)
            term3 = tf.expand_dims(term3, axis=1)

            total = term1 + term2 + term3 + b_p

            G = tf.nn.tanh(total)
            G = tf.reshape(G, [self.batch_size * self.Q, self.hidden_size])
            assert_dim("G", G, dim=1, expected_value=self.hidden_size)

            # Reshape before multiplication
            alpha = tf.nn.softmax(tf.matmul(G, W) + b)
            alpha = tf.reshape(alpha, [self.batch_size, 1, self.Q])

            attended = tf.matmul(alpha, self.H_q)
            inputs = tf.reshape(inputs, [self.batch_size, 1, self.hidden_size])
            z_i = tf.concat([inputs, attended], axis=2)
            z_i = tf.reshape(z_i, [self.batch_size, 2*self.hidden_size])

            output, states = self.cell(z_i, state)
            return output, (states.c, states.h)


class AnsPtrCell(RNNCell):
    """
    Note: I'm not actually certain we can do this in a batched fashion, as there are nodes of a different
    size per each batch item. If we use the static self._P, then we can predict outside of the bounds.
    """
    def __init__(self, Hr, hidden_size, mask=None):
        self._Hr = Hr

        shape = tf.shape(Hr)
        self._batch_size = shape[0]
        self._P = shape[1]

        if mask is None:
            self._mask = tf.ones([self._batch_size, self._P, 1], dtype=tf.float32)
        else:
            self._mask = mask

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

            # F = tanh(H_r V + h_(k)a W_a + b_a)
            s = batch_matmul(self._Hr, V)
            s += tf.reshape(tf.matmul(h_a, W_a), [-1, 1, self._hidden_size])
            s += b_a
            F = tf.nn.tanh(s)
            # F = tf.Print(F, [tf.shape(F)])
            assert_rank("F", F, expected_rank=3)

            # B = softmax(F_k v + c)
            s = batch_matmul(F, v) + c
            B_logits = s * self._mask
            B = tf.nn.softmax(s, dim=1)
            # B = tf.Print(B, [B], summarize=100)
            B = tf.reshape(B, [-1, 1, self._P])
            assert_rank("B", B, expected_rank=3)

            # h_(k)a = LSTM(B_k H_r, h_(k-1)a)
            cell_in = tf.matmul(B, self._Hr)
            cell_in = tf.reshape(cell_in, [-1, 2*self._hidden_size])
            _, state = self._cell(cell_in, state)

            B_logits = tf.reshape(s, [-1, self._P])
            return B_logits, (state.c, state.h)


######################################
#       Tests for the cells          #
######################################

def test_lstmwithatt():
    BATCH_SIZE, Q, HIDDEN_SIZE = [20, 300, 50]
    with tf.Session().as_default() as sess:
        with tf.variable_scope("testing"):
            H_q = tf.random_normal([BATCH_SIZE, Q, HIDDEN_SIZE], dtype=tf.float32)
            cell = LSTMCellWithAtt(H_q, HIDDEN_SIZE)
            H_p = tf.random_normal([1, HIDDEN_SIZE], dtype=tf.float32)
            state = cell.zero_state(1, dtype=tf.float32)
            out, state = cell(H_p, state)

        sess.run(tf.global_variables_initializer())
        output, h_f = sess.run([out, state])
        assert h_f.shape == []

def test_ansptr():
    BATCH_SIZE, P, HIDDEN_SIZE = [20, 300, 50]

    with tf.Session().as_default() as sess:
        with tf.variable_scope("testing"):
            Hr = tf.random_normal([BATCH_SIZE, P, 2*HIDDEN_SIZE], dtype=tf.float32)
            cell = AnsPtrCell(Hr, HIDDEN_SIZE)
            init_state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)

            B1, s1 = cell(None, init_state)
            tf.get_variable_scope().reuse_variables()
            B2, s2 = cell(None, s1)

        sess.run(tf.global_variables_initializer())
        B1, s1, B2, s2 = sess.run([B1, s1, B2, s2])
        assert np.allclose(np.sum(B1, axis=1), 1.0), "Beta1 rows must sum to 1"
        assert np.allclose(np.sum(B2, axis=1), 1.0), "Beta2 rows must sum to 1"


# Run the tests when called from command line
if __name__ == '__main__':
    test_lstmwithatt()
    test_ansptr()
    print("all tests passed!")
