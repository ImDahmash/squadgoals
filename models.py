from functools import lru_cache

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import BasicLSTMCell, GRUCell

from utils import *
from cells import *

class MatchLSTMModel(object):
    """Simple RNN model for SQuAD that uses BiLSTMs.
    """
    def __init__(self, config):

        # Setup configuration
        self._config = config

        # configure model variables
        # load GloVe embedding
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

        print("H_q shape", H_q.get_shape())

        with tf.variable_scope("encode_passage"):
            encode_cell = cell(self._config.hidden_size)
            H_p, _  = tf.nn.dynamic_rnn(encode_cell, contexts, self._clens, dtype=tf.float32)

            attention_cell = LSTMCellWithAtt(H_q, self._config.hidden_size)
            H_r, _ = tf.nn.bidirectional_dynamic_rnn(attention_cell, attention_cell,
                                                     H_p, sequence_length=self._clens,
                                                     dtype=tf.float32)
            H_r = tf.concat(H_r, axis=2)
            print("H_r shape: ", H_r.get_shape())

        with tf.variable_scope("answer_ptr"):
            # Perform decoding
            answer_cell = AnsPtrCell(H_r, self._config.hidden_size)
            state = answer_cell.zero_state(self._config.batch_size, dtype=tf.float32)

            B_s, state = answer_cell(None, state)
            tf.get_variable_scope().reuse_variables()
            B_e, _ = answer_cell(None, state)
            # ^ I'm relatively certain this is what we're supposed to be doing, take a look
            # at Figure 1(b) but this is how I read it.

        # Reshape these out
        B_s = tf.reshape(B_s, [self._config.batch_size, -1])
        B_e = tf.reshape(B_e, [self._config.batch_size, -1])

        loss = 0.0
        loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._starts, logits=B_s)
        loss += tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._ends, logits=B_e)
        self._loss = tf.Print(tf.reduce_mean(loss), [B_s, B_e, tf.shape(B_s), tf.shape(B_e)])

        # self._loss = tf.reduce_mean(self._loss)
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



class BiLSTMModel(object):
    """Simple RNN model for SQuAD that uses BiLSTMs.
    """
    def __init__(self, config):

        # Setup configuration
        self._config = config

        # configure model variables
        # load GloVe embedding
        self._embed = tf.Variable(self._load_embeddings(), name="embeddings")
        self._question = tf.placeholder(tf.int32, [None, None], "question_batch")
        self._context = tf.placeholder(tf.int32, [None, None], "context_batch")
        self._answer = tf.placeholder(tf.int32, [None, None], "labels_batch")
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
            cell = LSTMCell
        elif self._config.cell_type == "gru":
            cell = GRUCell

        # Get a representation of the questions
        with tf.variable_scope("encode_question"):
            encode_cell = MultiRNNCell([DropoutWrapper(cell(self._config.hidden_size), input_keep_prob=0.99)] * self._config.layers)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell,
                                                              questions, sequence_length=self._qlens,
                                                              dtype=tf.float32)
            state_fw, state_bw = states
            #final_q_state = tf.concat(h, 1)    # Size [batch_size, 2*hidden_size]
            #h_fw = (c[0], h[0])
            #h_bw = (c[1], h[1])

        # Print out debugging information

        with tf.variable_scope("encode_passage"):
            encode_cell = MultiRNNCell([cell(self._config.hidden_size)] * self._config.layers)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell,
                                                         contexts, sequence_length=self._clens,
                                                         initial_state_fw=state_fw,
                                                         initial_state_bw=state_bw,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)

        # Run the final set of outputs through an LSTM rnn that outputs one of 2 classes
        with tf.variable_scope("decode_answer"):
            decode_cell = MultiRNNCell([LSTMCell(2)] * 2)
            outputs, _ = tf.nn.dynamic_rnn(decode_cell, outputs, dtype=tf.float32, sequence_length=self._clens)

        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._answer, logits=outputs)
        self._loss = tf.reduce_mean(self._loss)
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
        feeds = {
            self._question: questions,
            self._context: contexts,
            self._answer: answers,
            self._qlens: qlens,
            self._clens: clens,
        }
        return feeds
