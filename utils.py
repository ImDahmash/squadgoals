"""
Simple utility functions that don't necessarily fit elsewhere.
"""

import os
from functools import wraps

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
        return tf.Session( ** kwargs)


def load_glove():
    try:
        return np.load("data/squad/glove.trimmed.100.npz")["glove"]
    except IOError as e:
        print("Error loading GloVe data: {}".format(e))


def minibatch_index_iterator(total_size, batch_size):
    """
    Generator that yields indexes to sample the minibatch from up to the total size.
    """
    order = np.random.permutation(total_size)
    for i in range(0, order.shape[0], batch_size):
        yield order[i:i + batch_size]

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

class Progress(object):
    def __init__(self, width=30):
        self._width = width
        self._dir = 1
        self._pos = 0
        self._displays = {} # List of things to display

    def tick(self, **kwargs):
        # Display new thing
        print("\r", end="")
        # New token position
        self._pos += self._dir
        if self._pos == self._width or self._pos == 0:
            self._dir *= -1
        if self._dir == 1:
            bar = "=" * self._pos + ">" + "." * (self._width - self._pos)
        else:
            bar = "=" * self._pos + "<" + "." * (self._width - self._pos)

        print(bar, end="")

