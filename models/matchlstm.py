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

        self._predictions = None
        self._loss = None
        self._train_op = None

    @lru_cache()
    def _load_embeddings(self):
        # Lazy compute the embedding matrix
        print("Loading GloVe vectors from {}".format(self._config.embed_path))
        return np.load(self._config.embed_path)

    def build_graph(self):
        # Add an embeddings layer
        questions = self._build_embedded(self._question)
        contexts = self._build_embedded(self._context)
        batch_size, P = tf.shape(questions)[0], tf.shape(contexts)[1]

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
            # Perform decoding
            # Create mask of dimension [batch_size, P, 1] that is used to scale the probabilities
            # and eliminate OOB predictions.
            # mask = tf.zeros([batch_size, P, 1])  # Size of the batch

            answer_cell = AnsPtrCell(H_r, self._config.hidden_size, mask=self._mask)
            state = answer_cell.zero_state(batch_size, dtype=tf.float32)

            # B_s and B_e are unscaled logits of the actual distribution.
            B_s, state = answer_cell(None, state)
            tf.get_variable_scope().reuse_variables()
            B_e, _ = answer_cell(None, state)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._starts, logits=B_s) \
                + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._ends, logits=B_e)
        self._loss = tf.reduce_mean(loss)
        self._train_op = AdamaxOptimizer(learning_rate=self._config.lr, beta1=0.9, beta2=0.999).minimize(self._loss)
        return self  # Return self to allow for chaining

    def train(self, questions, contexts, answers, qlens, clens, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens)
        _, loss = sess.run([self._train_op, self._loss], feed_dict=feeds)
        return loss

    def evaluate(self, questions, contexts, answers, qlens, clens, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens)
        loss = sess.run(self._loss, feed_dict=feeds)
        return loss

    def _build_embedded(self, ids, dtype=tf.float32):
        # Find pre-trained embeddings on disk
        embed = tf.nn.embedding_lookup(self._embed, ids)
        embed = tf.reshape(embed, [tf.shape(ids)[0], tf.shape(ids)[1], self._config.embed_dim])
        return tf.cast(embed, dtype=dtype)


    def _build_feeds(self, questions, contexts, answers, qlens, clens):
        # Build spans based on the answers
        batch_size = answers.shape[0]
        spans = np.zeros([batch_size, 2])
        for i in range(batch_size):
            line = answers[i]
            places = np.where(line != 0)[0].tolist()
            start, end = places[0], places[-1]
            spans[i] = np.array([start, end])

        # Mask out lengths
        mask = np.zeros([batch_size, contexts.shape[1], 1])
        for i in range(batch_size):
            start, end = int(spans[i, 0]), int(spans[i, 1])
            # print("start={} end={}".format(start, end))
            mask[i, start:end+1] = np.array([1])

        # import pdb; pdb.set_trace()
        # print(mask)

        feeds = {
            self._question: questions,
            self._context: contexts,
            self._starts: spans[:, 0],
            self._ends: spans[:, 1],
            self._mask: mask,
            self._qlens: qlens,
            self._clens: clens,
        }
        return feeds
