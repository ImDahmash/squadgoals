import logging
import sys
import math
from importlib import import_module

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from core import Config, SquadModel
from utils import minibatch_index_iterator

logging.getLogger().setLevel(logging.INFO)


def load_class(clsname):
    """
    Loads a model that is a subtype of SquadModel. @clsname@ gives the fully qualified Python
    package path to the model we wish to run.
    """
    try:
        pkg, cls = None, None
        if "." in clsname:
            components = clsname.split(".")
            pkg = ".".join(components[:-1])
            cls = components[-1]
            logging.info("loading package {}".format(pkg))
            p = import_module(pkg)
            logging.info("loading class {}".format(cls))
            return getattr(p, cls)
        else:
            logging.info("direct loading {}".format(cls))
            return globals()[clsname]
    except AttributeError or KeyError:
        logging.fatal("Error! Could not load class {}".format(clsname))
        sys.exit(-1)


def main(_):
    # Load and initialize the model
    model_class = load_class(tf.flags.FLAGS.model)
    if not issubclass(model_class, SquadModel):
        logging.fatal("Error! Given model {} is not an instance of core.SquadModel.".format(tf.flags.FLAGS.model))
        sys.exit(-1)

    # Configure the model and build the graph
    config = Config(dict(
        max_length=tf.flags.FLAGS.max_length,
        keep_prob=tf.flags.FLAGS.keep_prob,
        num_classes=2,
        embed_size=tf.flags.FLAGS.embed_size,
        hidden_size=tf.flags.FLAGS.hidden_size,
        cell_type=tf.flags.FLAGS.cell_type,
        save_dir=tf.flags.FLAGS.save_dir
    ))
    model = model_class()
    model.initialize_graph(config)

    logging.info("Loading training data...")
    train_data = np.load(tf.flags.FLAGS.train_path)
    question = train_data["question"]
    context = train_data["context"]
    answer = train_data["answer"]
    num_training_examples = context.shape[0]
    batch_size = tf.flags.FLAGS.batch_size
    num_batches = math.ceil(num_training_examples / batch_size)

    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(tf.flags.FLAGS.epochs):

            # Read a random batch of data
            epoch_num = epoch + 1
            # Store all of the training set in memory (yikes?)
            logging.info("epoch {} of {}".format(epoch_num, tf.flags.FLAGS.epochs))

            # We want to return a set of indexes
            batches = tqdm(minibatch_index_iterator(num_training_examples, batch_size), unit="batch", total=num_batches)
            for batch_idxs in batches:
                question_batch = question[batch_idxs]
                context_batch = context[batch_idxs]
                answer_batch = answer[batch_idxs]

                loss = model.train_batch(question_batch,
                                         context_batch,
                                         answer_batch)
                fmt_loss = "{:.6f}".format(loss)
                batches.set_postfix(loss=fmt_loss)

            logging.info("epoch={:03d} loss={}".format(epoch_num, loss))

            if epoch % tf.flags.FLAGS.checkpoint_freq == 0:
                # Perform evaluation on the smaller dev set.
                model.checkpoint(tf.flags.FLAGS.save_dir)


if __name__ == '__main__':
    # Setup arguments
    tf.flags.DEFINE_string("model", "models.baseline.BaselineModel", "full Python package path to a SquadModel")
    tf.flags.DEFINE_string("cell_type", "lstm", "type of RNN cell for training (either 'lstm' or 'gru'")

    tf.flags.DEFINE_integer("epochs", 50, "number of epochs of training")
    tf.flags.DEFINE_integer("checkpoint_freq", 1, "epochs of training between re-evaluating and saving model")
    tf.flags.DEFINE_integer("max_length", 250, "maximum length of sentences (in tokens)")
    tf.flags.DEFINE_integer("hidden_size", 20, "size of the hidden state for the RNN cell")
    tf.flags.DEFINE_integer("embed_size", 100, "dimensionality of embedding")
    tf.flags.DEFINE_integer("batch_size", 30, "size of minibatches")

    tf.flags.DEFINE_float("keep_prob", 0.99, "inverse of drop probability")

    tf.flags.DEFINE_string("train_path", "data/squad/train.npz", "Path to a .npz file that holds the training matrices")
    tf.flags.DEFINE_string("save_dir", "save", "path to save training results and model checkpoints")
    tf.flags.DEFINE_string("embed_path", "data/squad/glove.trimmed.100.npz", "path to npz file holding word embeddings")

    # Execute main() above, see tf.app documentation for details.
    tf.app.run()