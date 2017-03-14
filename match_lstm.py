from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from functools import lru_cache
import math
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, BasicLSTMCell, GRUCell, MultiRNNCell

from utils import minibatch_index_iterator, Progress


###############################################################
#       Code to setup the environment, configuration, etc.
###############################################################

"""
Global Variables                               #
"""
GLOVE_PATH = "data/squad/glove.squad.100d.npy"
TRAIN_PATH = "data/squad/train.npz"
VAL_PATH = "data/squad/val.npz"



"""
Configuration options:
"""
tf.flags.DEFINE_integer('hidden_size', 200, "Size of hidden states for encoder")
tf.flags.DEFINE_integer('batch_size', 32, 'size of mini-batches')
tf.flags.DEFINE_integer('embed_dim', 100, 'embedding dimension')
tf.flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
tf.flags.DEFINE_integer('layers', 2, 'number of hidden layers')

tf.flags.DEFINE_string('cell_type', 'lstm', "Cell type for RNN")
tf.flags.DEFINE_float('lr', 0.01, 'learning rate')

tf.flags.DEFINE_string('optim', 'adam', 'Optimizer, one of "adam", "adadelta", "sgd"')

tf.flags.DEFINE_integer('subset', 0, 'If > 0, only trains on a subset of the train data of given size')

tf.flags.DEFINE_string('embed_path', 'data/squad/glove.squad.100d.npy', "Path to a .npy file holding the GloVe vectors")
tf.flags.DEFINE_string('train_path', 'data/squad/train.npz', "Path to training data as an .npz file")
tf.flags.DEFINE_string('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file")
tf.flags.DEFINE_string('save_dir', 'save', 'directory to save model checkpoints after each epoch')

"""
Utilities
"""


def load_data(path):
    data = np.load(path)

    questions = data["question"]
    contexts = data["context"]
    answers = data["answer"]

    questions_lens = data["question_lens"]
    contexts_lens = data["context_lens"]

    return questions, contexts, answers, questions_lens, contexts_lens

def minibatch_indexes(maxidx, batch_size):
    # Perform random batch ordering
    order = np.random.permutation(maxidx)
    batch_idxs = []
    for i in range(0, maxidx, batch_size):
        batch_idxs.append(order[i:i + batch_size])
    return batch_idxs



def batch_matmul(xs, W):
    batch_size, m, n = tf.shape(xs)
    tf.reshape(xs, [batch_size * m, n])
    result = tf.matmul(xs, W)
    return tf.reshape(result, [batch_size, m, n])

###############################################################
#           Model Implementation
###############################################################

class LSTMCellWithAtt(tf.nn.rnn_cell.RNNCell):
    def __init__(self, hq, hidden_size):
        self.hq = hq
        self.hidden_size = hidden_size
        self.cell = BasicLSTMCell(self.hidden_size)

    @property
    def state_size(self):
        return self.hidden_size

    @property
    def output_size(self):
        return self.hidden_size

    def __call__(self, inputs, state, scope=None):
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            W_q = tf.get_variable("W_q", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W_p = tf.get_variable("W_p", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            W_r = tf.get_variable("W_r", [self.hidden_size, self.hidden_size], tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b_p = tf.get_variable("b_p", [self.hidden_size], tf.float32, initializer = tf.constant_initializer(0.0))
            b = tf.get_variable("b", [1], tf.float32, initializer = tf.constant_initializer(0.0))
            
            G = tf.nn.tanh(batch_matmul(self.hq, W_q) + tf.matmul(inputs, W_r) + b_p)
            alpha = tf.nn.softmax(batch_matmul(G, W)+ b)
            z = tf.matmul(tf.transpose(alpha), self.hq)
            z = tf.concat([inputs, z], axis=2)
            return self.cell(z, state)




class MatchLSTMModel(object):
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
            cell = BasicLSTMCell
        elif self._config.cell_type == "gru":
            cell = GRUCell

        # Get a representation of the questions
        with tf.variable_scope("encode_question"):
            encode_cell = cell(self._config.hidden_size)
            H_q, _ = tf.nn.dynamic_rnn(encode_cell, self._question,
                                       questions, sequence_length=self._qlens,
                                       dtype=tf.float32)

        with tf.variable_scope("encode_passage"):
            encode_cell = cell(self._config.hidden_size)
            H_p,  = tf.nn.dynamic_rnn(encode_cell, self._context,
                                      contexts, sequence_length=self._clens
                                      dtype=tf.float32)

            attention_cell = LSTMCellWithAtt(H_q, self._config.hidden_size)
            H_r, _ = tf.nn.bidirectional_rnn(attention_cell, attention_cell,
                                             H_p, sequence_length=self._clens,
                                             dtype=tf.float32)
            H_r = tf.concat(H_r, axis=2)



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


def main(_):
    with tf.Session().as_default() as sess:
        # Train the model
        config = tf.flags.FLAGS
        print("Configuration:")
        pprint(tf.flags.FLAGS.__dict__['__flags'], indent=4)
        model = BiLSTMModel(config).build_graph()

        sess.run(tf.global_variables_initializer())

        print("Number of parameters in the model: {}".format(len(tf.trainable_variables())))

        questions, contexts, answers, q_lens, c_lens = load_data(config.train_path)

        # Load validation set data to find test loss after each epoch
        val_qs, val_cs, val_as, val_q_lens, val_c_lens = load_data(config.val_path)

        num_examples = config.subset if config.subset > 0 else questions.shape[0]
        total_params = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        print("Total params: {}".format(total_params))


        # Setup saving
        saver = tf.train.Saver()
        save_path = os.path.join(config.save_dir, "model")
        if not tf.gfile.Exists(config.save_dir):
            tf.gfile.MakeDirs(config.save_dir)

        # Perform training pass
        epoch_losses = []
        for epoch in range(config.epochs):
            num_batches = math.ceil(num_examples / config.batch_size)
            losses = []

            # Create progress bar over this
            bar = Progress('Epoch {} of {}'.format(epoch + 1, config.epochs), steps=num_batches, width=20)
            for batch, idxs in enumerate(minibatch_index_iterator(num_examples, config.batch_size)):
                # Read batch_size indexes for constructing training batch
                qs = questions[idxs]
                cs = contexts[idxs]
                ans = answers[idxs]
                q_ls = q_lens[idxs]
                c_ls = c_lens[idxs]

                # Perform train step
                loss = model.train(qs, cs, ans, q_ls, c_ls)
                losses.append(loss)

                # Calculate some stats to print
                bar.tick(loss=loss, avg=np.average(losses), hi=max(losses), lo=min(losses))
            avg_loss = np.average(losses)
            epoch_losses.append(avg_loss)
            print("\n--- Epoch {} Average Train Loss: {:.7f}".format(epoch + 1, avg_loss))
            # Run validation, get validation loss
            val_loss = model.evaluate(val_qs, val_cs, val_as, val_q_lens, val_c_lens)
            print("  \ Validation Loss: {:.7f}".format(val_loss))
            # Save the model
            saver.save(sess, save_path, global_step=epoch)

        # Write the losses out to a file for later
        print("Saving statistics...")
        np.save("statistics.npz", epoch_losses=epoch_losses)


if __name__ == '__main__':
    tf.app.run(main=main)
