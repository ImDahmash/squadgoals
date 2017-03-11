import logging
from os.path import join as pjoin

import numpy as np
import tensorflow as tf
from tensorflow import nn
from tensorflow.contrib import rnn

from tensorflow.contrib.layers.python.layers import xavier_initializer

from core import SquadModel

logging.getLogger().setLevel(logging.INFO)


class BaselineModel(SquadModel):
    """
    BiLSTM SQuAD model, simple baseline.
    """

    def __init__(self, embeddings, train_embeddings=True):
        super(BaselineModel, self).__init__()
        self._question_placeholder = None
        self._passage_placeholder = None
        self._answer_placeholder = None
        self._mask_placeholder = None
        self._cell = None
        self._loss = None
        self._train_step = None
        self._model_output = None
        self._preds = None
        self._pretrained_embeddings = tf.Variable(embeddings, trainable=train_embeddings, name="embeddings")
        self._context_len_placeholder = None

    def _embedding(self, ids, embed_size):
        embed = tf.nn.embedding_lookup(self._pretrained_embeddings, ids)
        return tf.reshape(embed, [tf.shape(ids)[0], tf.shape(ids)[1], embed_size])

    def initialize_graph(self, config):
        self._question_placeholder = tf.placeholder(tf.int32, [None, None], "question_ids")
        self._passage_placeholder = tf.placeholder(tf.int32, [None, None], "passage_ids")
        self._answer_placeholder = tf.placeholder(tf.int32, [None, None], "answer_batch")
        self._mask_placeholder = tf.placeholder(tf.bool, [None, None], "mask_batch")
        self._context_len_placeholder = tf.placeholder(tf.int32, shape=[None], name="context_length")
        self._model_output = config.save_dir

        question_batch = self._embedding(self._question_placeholder, config.embed_size)
        passage_batch = self._embedding(self._passage_placeholder, config.embed_size)

        if config.cell_type == "lstm":
            cell = rnn.LSTMCell(config.hidden_size)
        elif config.cell_type == "gru":
            cell = rnn.GRUCell(config.hidden_size)
        else:
            raise ValueError("Invalid cell_type {}".format(config.cell_type))

        print(question_batch.get_shape())
        print(passage_batch.get_shape())

        # Okay fuck this shit so much
        # Use a stupid linear approximation
        with tf.variable_scope(self.__class__.__name__):
            W = tf.get_variable("W", [config.max_length, 2], tf.float32, xavier_initializer())
            b = tf.get_variable("b", [2], tf.float32, tf.constant_initializer(0.0))
            self._preds = tf.matmul(passage_batch, W) + b   # Should broadcast?
            self._loss = tf.constant(0.0)


    def train_batch(self, question_batch, passage_batch, answer_batch, question_lens, context_lens, glove_mat, sess=None):
        """
        TODO: replace feed_dict with whatever is supposed to be more efficient
        Link: https://www.tensorflow.org/programmers_guide/reading_data
        """

        if sess is None:
            sess = tf.get_default_session()

        # Create masking tensor
        mask = np.zeros_like(answer_batch, dtype='bool')
        for i, length in enumerate(context_lens):
            mask[i, 0:length] = True

        feeds = {
            self._question_placeholder: question_batch,
            self._passage_placeholder: passage_batch,
            self._answer_placeholder: answer_batch,
            self._mask_placeholder: mask,
            self._context_len_placeholder: context_lens,
        }

        _, loss = sess.run([self._train_step, self._loss], feed_dict=feeds)
        return loss

    def predict(self, question_batch, passage_batch, answer_batch, sess=None):
        """
        Predicts the (start, end) span representing the answer for the given question over the given passage.
        """
        if sess is None:
            sess = tf.get_default_session()

        feeds = {
            self._question_placeholder: question_batch,
            self._passage_placeholder: passage_batch,
            self._answer_placeholder: answer_batch,
        }

        return sess.run(self._loss, feed_dict=feeds)

    def checkpoint(self, save_dir, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        save_file = pjoin(self._model_output, "model.weights")
        saver = tf.train.Saver()
        saver.save(sess, save_file)

    def restore_from_checkpoint(self, save_dir, sess=None):
        if sess is None:
            sess = tf.get_default_session()

        restorer = tf.train.Saver()
        save_path = tf.train.latest_checkpoint(save_dir)
        logging.info("Restoring from {}".format(save_path))
        restorer.restore(sess, save_path)
