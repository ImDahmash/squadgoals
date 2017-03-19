from functools import lru_cache

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import GRUCell, DropoutWrapper, LSTMBlockCell

from adamax import AdamaxOptimizer
from cells import *
from utils import *
from bidirect import *


class MatchLSTMModel(object):
    """Model based on the MatchLSTM with Answer Pointer paper.
    """
    def __init__(self, config):

        # Setup configuration
        self._config = config

        # load GloVe embedding, don't train embedings, also don't save them
        self._embed = tf.constant(self._load_embeddings(), name="embeddings")
        self._question = tf.placeholder(tf.int32, [None, None], "question_batch")
        self._context = tf.placeholder(tf.int32, [None, None], "context_batch")
        self._starts = tf.placeholder(tf.int32, [None], "spans_batch")
        self._ends = tf.placeholder(tf.int32, [None], "spans_batch")
        self._qlens = tf.placeholder(tf.int32, [None], "question_lengths")
        self._clens = tf.placeholder(tf.int32, [None], "context_lengths")
        self._mask = tf.placeholder(tf.float32, [None, None, 1], "mask")
        self._keep_prob = tf.placeholder(tf.float32, [], "keep_prob")
        self._lr = tf.placeholder(tf.float32, [], "learning_rate")

        self.B_s = None
        self.B_e = None

        self._loss = None
        self._train_op = None
        self._grad_norm = None
        self._global_norm = 5.0 # Maximum norm at which point we want to clip.

    @lru_cache()
    def _load_embeddings(self):
        # Lazy compute the embedding matrix
        print("{}: Loading GloVe vectors from {}".format(self.__class__.__name__, self._config.embed_path))
        return np.load(self._config.embed_path)

    def build_graph(self):
        # Add an embeddings layer
        questions = self._build_embedded(self._question)
        contexts = self._build_embedded(self._context)
        batch_size = tf.shape(questions)[0]

        # Cell type based on config
        if self._config.cell_type == "lstm":
            cell = LSTMBlockCell
        elif self._config.cell_type == "gru":
            cell = GRUCell


        with tf.variable_scope("encode_question"):
            # Construct H_q from the paper, final shape should be [batch_size, Q, hidden_size]
            # Create encoding using the BiLSTM instead
            encode_cell = DropoutWrapper(cell(self._config.hidden_size), output_keep_prob=self._keep_prob)
            H_q, _ = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell, questions, self._qlens, dtype=tf.float32)
            H_q = tf.concat(H_q, 2)
            assert_rank("H_q", H_q, expected_rank=3)
            assert_dim("H_q", H_q, dim=2, expected_value=2*self._config.hidden_size)


        with tf.variable_scope("encode_passage"):
            encode_cell = DropoutWrapper(cell(self._config.hidden_size), output_keep_prob=self._keep_prob)
            H_p, _  = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell, contexts, self._clens, dtype=tf.float32)
            H_p = tf.concat(H_p, 2)
            assert_rank("H_p", H_p, expected_rank=3)
            assert_dim("H_p", H_p, dim=2, expected_value=2*self._config.hidden_size)

            # Calculate attention over question w.r.t. each token of the passage using the Match-LSTM cell.
            attention_cell = DropoutWrapper(LSTMCellWithAtt(H_q, 2*self._config.hidden_size), output_keep_prob=self._keep_prob)
            with tf.variable_scope("match_lstm"):
                H_r, _ = bidirectional_dynamic_rnn(attention_cell, attention_cell, H_p, sequence_length=self._clens, dtype=tf.float32)
                H_r = tf.concat(H_r, axis=2)
            assert_rank("H_r", H_r, expected_rank=3)
            assert_dim("H_r", H_r, dim=2, expected_value=4*self._config.hidden_size)

        with tf.variable_scope("answer_ptr"):
            answer_cell = AnsPtrCell(H_r, 2*self._config.hidden_size, mask=self._mask)
            initial_state = answer_cell.zero_state(batch_size, dtype=tf.float32)

            # B_s and B_e are unscaled logits of the actual distribution.
            B_s, s1 = answer_cell(None, initial_state)
            tf.get_variable_scope().reuse_variables()
            B_e, _ = answer_cell(None, s1)

        # Save these for predicting later, only calculated when we force them for .predict()
        self.B_s = tf.nn.softmax(B_s)
        self.B_e = tf.nn.softmax(B_e)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._starts, logits=B_s) \
                + tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._ends, logits=B_e)
        self._loss = tf.reduce_mean(loss)

        # COMMENT THIS OUT TO TURN OFF PRINTING
        # self._loss = tf.Print(self._loss, [self._keep_prob], summarize=300)

        # Returns the gradient norms
        optim = AdamaxOptimizer(learning_rate=self._lr)
        grads_and_vars = optim.compute_gradients(self._loss)
        gradients = [gv[0] for gv in grads_and_vars]
        variables = [gv[1] for gv in grads_and_vars]      # Uncomment this to do gradient clipping later

        # Perform gradient clipping
        gradients, _ = tf.clip_by_global_norm(gradients, self._global_norm)

        self._grad_norm = tf.global_norm(gradients)
        grads_and_vars = zip(gradients, variables)
        self._train_op = optim.apply_gradients(grads_and_vars)
        return self  # Return self to allow for chaining

    def train(self, questions, contexts, answers, qlens, clens, sess=None, norms=False, lr=0.01):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens, keep_prob=self._config.keep_prob, lr=lr)
        _, loss, grad_norm = sess.run([self._train_op, self._loss, self._grad_norm], feed_dict=feeds)
        if norms:
            return loss, grad_norm
        else:
            return loss

    def predict(self, questions, contexts, qlens, clens, sess=None):
        """
        Returns the probability distribution over the start and end tokens.
        """
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, None, qlens, clens, keep_prob=1.0)
        B_s, B_e = sess.run([self.B_s, self.B_e], feed_dict=feeds)
        # Return argmax across rows
        preds_start = np.argmax(B_s, axis=1)
        preds_end = np.argmax(B_e, axis=1)
        return preds_start, preds_end

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

    def _build_feeds(self, questions, contexts, answers, qlens, clens, keep_prob=1.0, lr=0.01):
        feeds = {
            self._question: questions,
            self._context: contexts,
            self._qlens: qlens,
            self._clens: clens,
            self._keep_prob: keep_prob,
            self._lr: lr,
        }
        batch_size = questions.shape[0]

        # Mask out lengths
        mask = np.zeros([batch_size, contexts.shape[1], 1])
        for i in range(batch_size):
            end = clens[i]
            mask[i, end:] = -1000.0
        feeds[self._mask] = mask

        if answers is not None:
            spans = np.zeros([batch_size, 2])
            for i in range(batch_size):
                line = answers[i]
                places = np.where(line != 0)[0].tolist()
                start, end = places[0], places[-1]
                spans[i] = np.array([start, end])


            feeds[self._starts] = spans[:, 0]
            feeds[self._ends] = spans[:, 1]
        return feeds
