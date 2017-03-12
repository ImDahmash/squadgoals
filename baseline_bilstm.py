from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from functools import wraps, lru_cache
import math
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell, MultiRNNCell

from tqdm import tqdm

from core import Config

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

# Default values for configuration parameters
CONFIGURABLE_PARAMS = [
      ('hidden_size', 200, "Size of hidden states for encoder")
    , ('batch_size', 32, 'size of mini-batches')
    , ('embed_dim', 100, 'embedding dimension')
    , ('epochs', 10, 'number of epochs for training')

    , ('cell_type', 'lstm', "Cell type for RNN")
    , ('lr', 0.01, 'learning rate')
    , ('optim', 'adam', 'Optimizer, one of "adam", "adadelta", "sgd"')

    , ('subset', 0, 'If > 0, only trains on a subset of the train data of given size')

    , ('embed_path', 'data/squad/glove.squad.100d.npy', "Path to a .npy file holding the GloVe vectors")
    , ('train_path', 'data/squad/train.npz', "Path to training data as an .npz file")
    , ('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file")
    , ('save_dir', 'save', 'directory to save model checkpoints after each epoch')
]

# Default configuration object
DEFAULT_CONFIG = {
    name: default for name, default, _ in CONFIGURABLE_PARAMS
}

print("DEFAULT_CONFIG")
pprint(DEFAULT_CONFIG)

def setup_args():
    # setup arguments
    for name, default, doc in CONFIGURABLE_PARAMS:
        if isinstance(default, int):
            func = tf.flags.DEFINE_integer
        elif isinstance(default, str):
            func = tf.flags.DEFINE_string
        elif isinstance(default, float):
            func = tf.flags.DEFINE_float
        elif isinstance(default, bool):
            func = tf.flags.DEFINE_boolean

        func(name, default, doc)


def compute_once(expensive):
    """
    Decorator for once-evaluated function. Python 3.x only (use of 'nonlocal' keyword)
    """
    computed = False
    value = None
    @wraps(expensive)
    def inner(*args, **kwargs):
        nonlocal computed
        nonlocal value
        if not computed:
            value = expensive(*args, **kwargs)
            computed = True
        return value
    return inner



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



###############################################################
#           Model Implementation
###############################################################

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
            encode_cell = MultiRNNCell([cell(self._config.hidden_size)] * 2)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell,
                                                              questions, sequence_length=self._qlens,
                                                              dtype=tf.float32)
            state_fw, state_bw = states
            #final_q_state = tf.concat(h, 1)    # Size [batch_size, 2*hidden_size]
            #h_fw = (c[0], h[0])
            #h_bw = (c[1], h[1])

        # Print out debugging information

        with tf.variable_scope("encode_passage"):
            encode_cell = MultiRNNCell([cell(self._config.hidden_size)] * 2)
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

        # loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._answer, logits=outputs)
        # Transform this hidden state into a vector of the desired size
        # self._loss = tf.Print(tf.reduce_mean(loss), [tf.shape(outputs)])
        # total = tf.reduce_sum(self._answer)
        real = tf.nn.softmax(outputs)[:, :, 1]
        diff = tf.cast(self._answer, tf.float32) - real
        self._loss = tf.reduce_mean(tf.square(diff))#tf.Print(tf.reduce_mean(loss), [tf.reduce_sum(tf.square(diff))], summarize=1000)#[total, matching, matching / tf.cast(total, tf.float32)])
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._config.lr).minimize(self._loss)

        return self # Return self to allow for chaining

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
        config = tf.flags.FLAGS.__dict__
        config.update(DEFAULT_CONFIG)
        config = Config(config)
        model = BiLSTMModel(config).build_graph()

        sess.run(tf.global_variables_initializer())

        questions, contexts, answers, q_lens, c_lens = load_data(config.train_path)

        # Load validation set data to find test loss after each epoch
        val_qs, val_cs, val_as, val_q_lens, val_c_lens = load_data(config.val_path)

        num_examples = questions.shape[0] if config.subset <= 0 else config.subset

        # Setup saving
        saver = tf.train.Saver()
        save_path = os.path.join(config.save_dir, "model")
        if not tf.gfile.Exists(config.save_dir):
            tf.gfile.MakeDirs(config.save_dir)

        # Perform training pass
        epoch_losses = []
        for epoch in range(config.epochs):
            batch_idxs = minibatch_indexes(num_examples, config.batch_size)
            num_batches = math.ceil(num_examples / config.batch_size)
            losses = []

            print("Epoch: {} / {}".format(epoch + 1, config.epochs))
            for batch in range(num_batches):
                # Read batch_size indexes for constructing training batch
                idxs = batch_idxs[batch]
                qs = questions[idxs]
                cs = contexts[idxs]
                ans = answers[idxs]
                q_ls = q_lens[idxs]
                c_ls = c_lens[idxs]

                # Perform train step
                tic = time.time()
                loss = model.train(qs, cs, ans, q_ls, c_ls)
                toc = time.time()
                losses.append(loss)
                print("\rBatch {} of {} === Loss: {:.7f}     Time: {:.4f}s          ".format(batch+1, num_batches, loss, toc - tic), end="")
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
    setup_args()
    tf.app.run(main=main)
