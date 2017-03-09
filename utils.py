"""
Simple utility functions that don't necessarily fit elsewhere.
"""

import os

import numpy as np
import tensorflow as tf


def try_jitted_session(**kwargs):
    """
    Creates a session that uses the XLA JIT compiler for NVIDIA platforms (i.e. the Azure machines).
    :return: A tf.Session that has JIT compilation of XLA enabled if available, otherwise a vanilla session.
    """
    if os.uname().sysname == "Linux":
        # Config to turn on JIT compilation
        config = tf.ConfigProto()
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        return tf.Session(config=config)
    else:
        return tf.Session(**kwargs)


def load_glove():
    try:
        return np.load("data/squad/glove.trimmed.100.npz")["glove"]
    except IOError as e:
        print("Error loading GloVe data: {}".format(e))


def load_train_data():
    """
    Loads training data ready for consumption by training procedure.
    """
    pass

def minibatch_index_iterator(total_size, batch_size):
    """
    Generator that yields indexes to sample the minibatch from up to the total size.
    """
    order = np.random.permutation(total_size)
    for i in range(0, order.shape[0], batch_size):
        yield order[i:i+batch_size]
