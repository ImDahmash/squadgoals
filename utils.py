"""
Simple utility functions that don't necessarily fit elsewhere.
"""

import time
import math
from functools import wraps

import numpy as np
# import tensorflow as tf


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

def batch_matmul(xs, W):
    import tensorflow as tf
    shape = tf.shape(xs)
    W_shape = tf.shape(W)
    xs = tf.reshape(xs, [shape[0] * shape[1], shape[2]])
    result = tf.matmul(xs, W)
    return tf.reshape(result, [shape[0], shape[1], W_shape[1]])


class Progress(object):
    def __init__(self, title="", width=30, steps='unknown'):
        self._title = title
        self._width = width
        self._steps = steps
        self._pos = -1
        self._last_tick = None

    def tick(self, **kwargs):
        self._pos += 1
        title = "{} ({} of {}):  ".format(self._title, self._pos, self._steps)

        # Provide estimate of time between ticks
        now = time.time()
        if self._last_tick is not None:
            delta = now - self._last_tick
            eta = (self._steps - self._pos) * delta
            kwargs['time'] = "{:.2f}s".format(delta)
            kwargs['hz'] = "{:.2f}it/s".format(1.0 / delta)
            kwargs['eta'] = "{:02d}:{:02d}".format(math.floor(eta / 60), int(eta % 60))
        self._last_tick = now

        pairs = []
        for k, v in kwargs.items():
            # Truncate floats
            if isinstance(v, float) or isinstance(v, np.float) or isinstance(v, np.float32) or isinstance(v, np.float64):
                v = "{:.4f}".format(v)
            pairs.append("{}={}".format(k, v))
        extra_info = '   '.join(pairs)

        print("\r" +  " " * 200, end="\r")
        text = title + "\t" + extra_info
        print(text, end="\r")

        if self._pos == self._steps:
            print()
