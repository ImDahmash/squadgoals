import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn

from core import SquadModel, EncoderDecoderModel


"""
P: He died on January 7, 1995 in a winter car crash
Q: When did he die?
A: "died January 7, 1995 winter" etc.
"""


class BaselineModel(SquadModel, EncoderDecoderModel):
    """
    BiLSTM SQuAD model, simple baseline.
    """

    def __init__(self):
        super(BaselineModel, self).__init__()
        self._question_placeholder = None
        self._passage_placeholder = None
        self._answer_placeholder = None
        self._range = None
        self._cell = None
        self._loss_op = None

    def initialize(self, config):
        self._question_placeholder = tf.placeholder(tf.float32, [None, None, config.embedding_size],
                                                    "question_embedded_batch")
        self._passage_placeholder = tf.placeholder(tf.float32, [None, None, config.embedding_size],
                                                   "passage_embedded_batch")
        self._answer_placeholder = tf.placeholder(tf.int32, [None, None, config.embedding_size], "answer_batch")
        self._range = tf.placeholder(tf.float32, [None, 2], "answer_range_batch")

        cell = rnn.LSTMCell if config.cell_type == "lstm" else rnn.GRUCell
        self._cell = cell(config.hidden_size)

        # Encode the question and passage, returns hidden states h_q, h_p and a set of the intermediate
        # hidden states output for each timestep.
        h_q, h_p, q_outs, p_outs = self.encode(self._question_placeholder, self._passage_placeholder)

        h = tf.concat([h_q, h_p], 2)

        # Run through a new LSTM that has two hidden outputs
        last_cell = rnn.LSTMCell(2)
        states, _ = nn.dynamic_rnn(last_cell, h)
        best_states = tf.ones_like(states, tf.float32)

        # Cross entropy with logits, no need to softmax directly
        loss = nn.softmax_cross_entropy_with_logits(best_states, states)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

        self._loss_op = optimizer.minimize(loss)

    def encode(self, question_batch, paragraph_batch):

        question_batch = tf.cast(question_batch, tf.float32)
        paragraph_batch = tf.cast(paragraph_batch, tf.float32)

        # Encode the question first
        # h_q and h_p are tuples of the form (hidden_state_forward, hidden_state_backward).
        # We stack them vertically to form our "final" hidden states.
        print("cell", self._cell)
        q_outs, h_q = nn.bidirectional_dynamic_rnn(
            self._cell, self._cell, question_batch, dtype=tf.float32,
            sequence_length=[100 for _ in range(100)])
        h_q = tf.concat(h_q, 2)
        p_outs, h_p = nn.bidirectional_dynamic_rnn(
            self._cell, self._cell, paragraph_batch, initial_state_fw=h_q,
            sequence_length=[100 for _ in range(100)])
        h_p = tf.concat(h_p, 2)

        # Same for our intermediate outputs
        q_outs = tf.concat(q_outs, 2)
        p_outs = tf.concat(p_outs, 2)

        # Return ALL THE THINGS!
        return h_q, h_p, q_outs, p_outs

    def decode(self, h_q, h_p, attention):
        """
        Take the two, calculates the attention vector, etc.
        """
        # THIS DOES NOTHING HOORAY
        pass

    def train_batch(self, sess, question_ids, passage_ids, answer_start, answer_end):
        # Construct a tensor that is 1 for positions that is between answer_start and answer_end
        answer_batch = tf.zeros_like(passage_ids)

        feed_dict = {
            self._question_placeholder: question_ids,
            self._passage_placeholder: passage_ids,
            self._answer_placeholder: answer_batch,
        }

        loss = sess.run([self._loss_op], feed_dict=feed_dict)


    def predict(self, question_ids, passage_ids):
        """
        Runs the neural network in a forward pass to predict tokens.
        :param question_ids:
        :param passage_ids:
        :return:
        """
        pass
