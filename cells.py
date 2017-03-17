import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell, BasicLSTMCell

from utils import *


class LSTMCellWithAtt(RNNCell):
    def __init__(self, Hq, hidden_size):
        self._Hq = Hq
        self._hidden_size = hidden_size
        self._cell = BasicLSTMCell(hidden_size)

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def state_size(self):
        return (self._hidden_size, self._hidden_size)

    def __call__(self, hp, state, scope=None):
        # Extract h_r as second element of state, discard the cell c_r
        _, hr = state

        with tf.variable_scope(scope or type(self).__name__):
            with tf.variable_scope("weights", dtype=tf.float32):
                Wq = tf.get_variable("Wq", [self._hidden_size, self._hidden_size], initializer=tf.contrib.layers.xavier_initializer())
                Wr = tf.get_variable("Wr", [self._hidden_size, self._hidden_size], initializer=tf.contrib.layers.xavier_initializer())
                Wp = tf.get_variable("Wp", [self._hidden_size, self._hidden_size], initializer=tf.contrib.layers.xavier_initializer())
                Wg = tf.get_variable("Wg", [self._hidden_size, 1], initializer=tf.contrib.layers.xavier_initializer())
            with tf.variable_scope("biases", initializer=tf.constant_initializer(0.0), dtype=tf.float32):
                bp = tf.get_variable("bp", [self._hidden_size])
                b = tf.get_variable("b", [])

            qterm = batch_matmul(self._Hq, Wq)
            rpterm = tf.matmul(hp, Wp)
            rpterm += tf.matmul(hr, Wr)
            rpterm += bp

            G = tf.nn.tanh(qterm + tf.expand_dims(rpterm, 1))
            alpha = tf.nn.softmax(batch_matmul(G, Wg) + b, dim=1)
            alpha = tf.transpose(alpha, [0, 2, 1])

            att = tf.matmul(alpha, self._Hq)
            z = tf.concat([hp, tf.squeeze(att, axis=1)], 1)

            output, statetup = self._cell(z, state)
            return output, (statetup.c, statetup.h)


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
            with tf.variable_scope("weights"):
                V = tf.get_variable("V", shape=[2*self._hidden_size, self._hidden_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                W_a = tf.get_variable("W_a", shape=[self._hidden_size, self._hidden_size], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
                v = tf.get_variable("v", shape=[self._hidden_size, 1], dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("biases", initializer=tf.constant_initializer(0.0)):
                b_a = tf.get_variable("b_a", shape=[1, self._hidden_size], dtype=tf.float32)
                c = tf.get_variable("c", shape=[1], dtype=tf.float32)

            # F = tanh(H_r V + h_(k)a W_a + b_a)
            s = batch_matmul(self._Hr, V)
            s += tf.reshape(tf.matmul(h_a, W_a), [-1, 1, self._hidden_size])
            s += b_a
            F = tf.nn.tanh(s)
            assert_rank("F", F, expected_rank=3)

            # B = softmax(F_k v + c)
            s = batch_matmul(F, v)
            s += c
            B_logits = tf.squeeze(s + self._mask)   # Should add -1000 to all the padded positions
            B = tf.nn.softmax(s, dim=1)
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
    BATCH_SIZE, Q, HIDDEN_SIZE = [10, 300, 50]
    with tf.Session().as_default() as sess:
        with tf.variable_scope("testing"):
            H_q = tf.random_normal([BATCH_SIZE, Q, HIDDEN_SIZE], dtype=tf.float32)
            cell = LSTMCellWithAtt(H_q, HIDDEN_SIZE)
            h_p = tf.random_normal([BATCH_SIZE, HIDDEN_SIZE], dtype=tf.float32)
            state = cell.zero_state(BATCH_SIZE, dtype=tf.float32)
            out, state = cell(h_p, state)

        sess.run(tf.global_variables_initializer())
        output, h_f = sess.run([out, state])
        assert h_f[1].shape == (BATCH_SIZE, HIDDEN_SIZE), "Actual shape: {}".format(h_f[1].shape)

def test_ansptr():
    BATCH_SIZE, P, HIDDEN_SIZE = [20, 300, 50]

    def softmax(x):
        x = np.exp(x)
        x /= np.sum(x, axis=1, keepdims=True)
        return x

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
        assert np.allclose(np.sum(softmax(B1), axis=1), 1.0), "Beta1 rows must sum to 1"
        assert np.allclose(np.sum(softmax(B2), axis=1), 1.0), "Beta2 rows must sum to 1"


# Run the tests when called from command line
if __name__ == '__main__':
    test_lstmwithatt()
    test_ansptr()
    print("all tests passed!")
