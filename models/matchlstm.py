from functools import lru_cache

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell

from adamax import AdamaxOptimizer
from cells import *
from utils import *


class MatchLSTMModel(object):
    """Model based on the MatchLSTM with Answer Pointer paper.
    """
    def __init__(self, config):

        # Setup configuration
        self._config = config

        # configure model variables
        # load GloVe embedding, don't train embedings
        self._embed = tf.Variable(self._load_embeddings(), name="embeddings", trainable=False)
        self._question = tf.placeholder(tf.int32, [None, None], "question_batch")
        self._context = tf.placeholder(tf.int32, [None, None], "context_batch")
        self._starts = tf.placeholder(tf.int32, [None], "spans_batch")
        self._ends = tf.placeholder(tf.int32, [None], "spans_batch")
        self._qlens = tf.placeholder(tf.int32, [None], "question_lengths")
        self._clens = tf.placeholder(tf.int32, [None], "context_lengths")
        self._mask = tf.placeholder(tf.float32, [None, None, 1], "mask")

        self.B_s = None
        self.B_e = None

        self._loss = None
        self._train_op = None
        self._grad_norm = None

    @lru_cache()
    def _load_embeddings(self):
        # Lazy compute the embedding matrix
        print("Loading GloVe vectors from {}".format(self._config.embed_path))
        return np.load(self._config.embed_path)

    def build_graph(self):
        # Add an embeddings layer
        questions = self._build_embedded(self._question)
        contexts = self._build_embedded(self._context)
        batch_size = tf.shape(questions)[0]

        # Cell type based on config
        if self._config.cell_type == "lstm":
            cell = BasicLSTMCell
        elif self._config.cell_type == "gru":
            cell = GRUCell

        # Get a representation of the questions
        with tf.variable_scope("encode_question"):
            encode_cell = cell(self._config.hidden_size)
            H_q, _ = tf.nn.dynamic_rnn(encode_cell, questions, self._qlens, dtype=tf.float32)

        with tf.variable_scope("encode_passage"):
            encode_cell = cell(self._config.hidden_size)
            H_p, _  = tf.nn.dynamic_rnn(encode_cell, contexts, self._clens, dtype=tf.float32)

            attention_cell = LSTMCellWithAtt(H_q, self._config.hidden_size)
            H_r, _ = tf.nn.bidirectional_dynamic_rnn(attention_cell, attention_cell,
                                                     H_p, sequence_length=self._clens,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r, axis=2)
            assert_dim("H_r", H_r, dim=2, expected_value = 2*self._config.hidden_size)

        with tf.variable_scope("answer_ptr"):
            answer_cell = AnsPtrCell(H_r, self._config.hidden_size, mask=self._mask)
            state = answer_cell.zero_state(batch_size, dtype=tf.float32)

            # B_s and B_e are unscaled logits of the actual distribution.
            B_s, state = answer_cell(None, state)
            tf.get_variable_scope().reuse_variables()
            B_e, _ = answer_cell(None, state)

        # Save these for predicting later, only calculated when we force them for .predict()
        self.B_s = tf.nn.softmax(B_s)
        self.B_e = tf.nn.softmax(B_e)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._starts, logits=B_s) \
                + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._ends, logits=B_e)
        self._loss = tf.reduce_mean(loss)

        optim = AdamaxOptimizer(learning_rate=self._config.lr, beta1=0.9, beta2=0.999)
        grads_and_vars = optim.compute_gradients(self._loss)
        gradients = [gv[0] for gv in grads_and_vars]
        # variables = [gv[1] for gv in grads_and_vars]
        self._grad_norm = tf.global_norm(gradients)
        self._train_op = optim.apply_gradients(grads_and_vars)
        return self  # Return self to allow for chaining

    def train(self, questions, contexts, answers, qlens, clens, sess=None, norms=False):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens)
        _, loss, grad_norm = sess.run([self._train_op, self._loss, self._grad_norm], feed_dict=feeds)
        if norms:
            return loss, grad_norm
        else:
            return loss

    def predict(self, questions, contexts, qlens, clens, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, None, qlens, clens)
        B_s_logits, B_e_logits = sess.run([self.B_s, self.B_e], feed_dict=feeds)

    def evaluate(self, questions, contexts, answers, qlens, clens, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens)
        loss = sess.run(self._loss, feed_dict=feeds)
        return loss

    def _build_embedded(self, ids, dtype=tf.float32):
        embed = tf.nn.embedding_lookup(self._embed, ids)
        embed = tf.reshape(embed, [tf.shape(ids)[0], tf.shape(ids)[1], self._config.embed_dim])
        return tf.cast(embed, dtype=dtype)

    def _build_feeds(self, questions, contexts, answers, qlens, clens):
        feeds = {
            self._question: questions,
            self._context: contexts,
            self._qlens: qlens,
            self._clens: clens,
        }
        batch_size = answers.shape[0]

        if answers is not None:
            spans = np.zeros([batch_size, 2])
            for i in range(batch_size):
                line = answers[i]
                places = np.where(line != 0)[0].tolist()
                start, end = places[0], places[-1]
                spans[i] = np.array([start, end])

            # Mask out lengths
            mask = np.zeros([batch_size, contexts.shape[1], 1])
            for i in range(batch_size):
                end = clens[i]
                mask[i, end:] = -1000.0

            feeds[self._starts] = spans[:, 0]
            feeds[self._ends] = spans[:, 1]
            feeds[self._mask] = mask
        return feeds
