from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from functools import wraps, lru_cache
import math
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, GRUCell

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
    ('hidden_size', 200, "Size of hidden states for encoder"),
    ('batch_size', 32, 'size of mini-batches'),
    ('embed_dim', 100, 'embedding dimension'),
    ('epochs', 10, 'number of epochs for training'),

    ('cell_type', 'lstm', "Cell type for RNN"),
    ('lr', 0.01, 'learning rate'),
    ('optim', 'adam', 'Optimizer, one of "adam", "adadelta", "sgd"'),

    ('max_question', 40, 'maximum length of question (in tokens)'),
    ('max_context', 100, 'maximum length of passage (in tokens)'),

    ('embed_path', 'data/squad/glove.squad.100d.npy', "Path to a .npy file holding the GloVe vectors"),
    ('train_path', 'data/squad/train.npz', "Path to training data as an .npz file"),
    ('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file"),
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
    Once-evaluated function. Python 3.x only (use of 'nonlocal' keyword)
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
        self._question = tf.placeholder(tf.int32, [None, self._config.max_question], "question_batch")
        self._context = tf.placeholder(tf.int32, [None, self._config.max_context], "context_batch")
        self._answer = tf.placeholder(tf.int32, [None, self._config.max_context], "labels_batch")
        self._question_mask = tf.placeholder(tf.bool, [None, self._config.max_question], "question_mask")
        self._context_mask = tf.placeholder(tf.bool, [None, self._config.max_context], "context_mask")
        self._qlens = tf.placeholder(tf.int32, [None], "question_lengths")
        self._clens = tf.placeholder(tf.int32, [None], "context_lengths")

        self._predictions = None
        self._loss = None
        self._train_op = None

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
            encode_cell = cell(self._config.hidden_size)
            outputs, states = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell,
                                                              questions, sequence_length=self._qlens,
                                                              dtype=tf.float32)
            state_fw, state_bw = states
            #final_q_state = tf.concat(h, 1)    # Size [batch_size, 2*hidden_size]
            #h_fw = (c[0], h[0])
            #h_bw = (c[1], h[1])

        # Print out debugging information

        with tf.variable_scope("encode_passage"):
            encode_cell = cell(self._config.hidden_size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(encode_cell, encode_cell,
                                                         contexts, sequence_length=self._clens,
                                                         initial_state_fw=state_fw,
                                                         initial_state_bw=state_bw,
                                                         dtype=tf.float32)
            outputs = tf.concat(outputs, axis=2)

        # Run the final set of outputs through an LSTM rnn that outputs one of 2 classes
        with tf.variable_scope("decode_answer"):
            decode_cell = LSTMCell(2)
            outputs, _ = tf.nn.dynamic_rnn(decode_cell, outputs, dtype=tf.float32, sequence_length=self._clens)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self._answer, logits=outputs)
        # Transform this hidden state into a vector of the desired size
        # self._loss = tf.Print(tf.reduce_mean(loss), [tf.shape(outputs)])
        total = tf.reduce_sum(self._answer)
        matching = tf.reduce_sum(tf.cast(self._answer, tf.float32) * tf.nn.softmax(outputs)[:, :, 1])
        self._loss = tf.Print(tf.reduce_mean(loss), [total, matching])
        self._train_op = tf.train.AdamOptimizer(learning_rate=self._config.lr).minimize(self._loss)

        return self # Return self to allow for chaining


    @lru_cache()
    def _load_embeddings(self):
        # Lazy compute the embedding matrix
        print("Loading GloVe vectors from {}".format(self._config.embed_path))
        return np.load(self._config.embed_path)

    def train(self, questions, contexts, answers, qlens, clens, sess=None):
        if sess is None:
            sess = tf.get_default_session()
        feeds = self._build_feeds(questions, contexts, answers, qlens, clens)
        _, loss = sess.run([self._train_op, self._loss], feed_dict=feeds)
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

def main(_):
    with tf.Session().as_default() as sess:
        # Train the model
        config = tf.flags.FLAGS.__dict__
        config.update(DEFAULT_CONFIG)
        config = Config(config)
        model = BiLSTMModel(config).build_graph()

        sess.run(tf.global_variables_initializer())

        questions, contexts, answers, q_lens, c_lens = load_data(config.train_path)

        num_examples = questions.shape[0]

        # Perform training pass
        for epoch in range(config.epochs):
            batch_idxs = minibatch_indexes(num_examples, config.batch_size)
            num_batches = math.ceil(num_examples / config.batch_size)
            print("Epoch: {} / {}".format(epoch + 1, config.epochs))
            losses = []
            for batch in range(num_batches):
                idxs = batch_idxs[batch]
                qs = questions[idxs]
                cs = contexts[idxs]
                ans = answers[idxs]
                q_ls = q_lens[idxs]
                c_ls = c_lens[idxs]
                loss = model.train(qs, cs, ans, q_ls, c_ls)
                losses.append(loss)
                time.sleep(0.5)
                print("\rBatch {} of {} === Loss: {:.7f}               ".format(batch+1, num_batches, loss), end="")
            avg_loss = np.average(losses)
            print("\nEpoch {} Average Loss: {:.7f}".format(epoch + 1, avg_loss))

if __name__ == '__main__':
    setup_args()
    tf.app.run(main=main)
