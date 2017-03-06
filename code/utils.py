"""
Simple utility functions that don't necessarily fit elsewhere.
"""

import os

import tensorflow as tf


def try_jitted_session(**kwargs):
    tf.VERSION
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
