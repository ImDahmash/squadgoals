from functools import lru_cache

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell

from utils import *
from cells import *

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

        with tf.variable_scope("answer_ptr"):
            # Perform decoding
            answer_cell = AnsPtrCell(H_r, self._config.hidden_size)
            state = answer_cell.zero_state(tf.shape(H_r)[0], dtype=tf.float32)

            B_s, state = answer_cell(None, state)
            tf.get_variable_scope().reuse_variables()
            B_e, _ = answer_cell(None, state)

        # Reshape these out
        B_s = tf.reshape(B_s, [tf.shape(questions)[0], -1])
        B_e = tf.reshape(B_e, [tf.shape(questions)[0], -1])
        B_s = tf.Print(B_s, [tf.shape(B_s), tf.shape(B_e), tf.shape(self._starts), tf.shape(self._ends)])

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._starts, logits=B_s) \
                + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._ends, logits=B_e)

        # loss = tf.Print(loss, [B_e], summarize=100)
        # self._loss = tf.Print(tf.reduce_mean(loss), [B_s, B_e, tf.shape(B_s), tf.shape(B_e)], summarize=100)
        self._loss = tf.reduce_mean(loss)

        self._train_op = tf.train.AdamOptimizer(learning_rate=self._config.lr).minimize(self._loss)

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

        feeds = {
            self._question: questions,
            self._context: contexts,
            self._starts: spans[:, 0],
            self._ends: spans[:, 1],
            self._qlens: qlens,
            self._clens: clens,
        }
        return feeds
