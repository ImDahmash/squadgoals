import logging
import math
import sys
from importlib import import_module

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
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


def train_model(model):
    logging.info("Loading training data...")
    train_data = np.load(tf.flags.FLAGS.data_path)
    question = train_data["question"]
    context = train_data["context"]
    answer = train_data["answer"]
    question_lens = train_data["question_lens"]
    context_lens = train_data["context_lens"]
    num_training_examples = context.shape[0]
    batch_size = tf.flags.FLAGS.batch_size
    num_batches = math.ceil(num_training_examples / batch_size)

    # Load full GloVe matrix
    glove_mat = np.load(tf.flags.FLAGS.embed_path)

    # Create save_dir for checkpointing if it does not already exist
    if not gfile.Exists(tf.flags.FLAGS.save_dir):
        gfile.MakeDirs(tf.flags.FLAGS.save_dir)

    with tf.Session().as_default() as sess:
        sess.run(tf.global_variables_initializer())

        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        logging.info("Number of trainable parameters: {}".format(num_params))

        for epoch in range(tf.flags.FLAGS.epochs):

            # Read a random batch of data
            epoch_num = epoch + 1
            # Store all of the training set in memory (yikes?)
            logging.info("epoch {} of {}".format(epoch_num, tf.flags.FLAGS.epochs))

            # We want to return a set of indexes
            batches = tqdm(minibatch_index_iterator(num_training_examples, batch_size), unit="batch", total=num_batches)
            losses = []
            for batch_idxs in batches:
                question_batch = question[batch_idxs]
                context_batch = context[batch_idxs]
                answer_batch = answer[batch_idxs]
                c_lens = context_lens[batch_idxs]
                q_lens = question_lens[batch_idxs]

                loss = model.train_batch(question_batch,
                                         context_batch,
                                         answer_batch,
                                         q_lens,
                                         c_lens,
                                         glove_mat)
                losses.append(loss)
                fmt_loss = "{:.6f}".format(loss)
                avg_loss = np.sum(losses) / len(losses)
                fmt_avg_loss = "{:.6f}".format(avg_loss)
                batches.set_postfix(loss=fmt_loss, avg=fmt_avg_loss)

            loss = np.sum(losses) / len(losses)
            logging.info("epoch={:03d} loss={}".format(epoch_num, loss))

            if epoch % tf.flags.FLAGS.checkpoint_freq == 0:
                # Perform evaluation on the smaller dev set.
                model.checkpoint(tf.flags.FLAGS.save_dir)


def eval_model(model):
    with tf.Session().as_default() as sess:
        model.restore_from_checkpoint(tf.flags.FLAGS.save_dir)
        logging.info("Loading training data...")
        eval_data = np.load(tf.flags.FLAGS.data_path)
        question = eval_data["question"]
        context = eval_data["context"]
        answer = eval_data["answer"]
        num_training_examples = context.shape[0]
        batch_size = tf.flags.FLAGS.batch_size

        # Create save_dir for checkpointing if it does not already exist
        if not gfile.Exists(tf.flags.FLAGS.save_dir):
            gfile.MakeDirs(tf.flags.FLAGS.save_dir)

        sess.run(tf.global_variables_initializer())
        batches = minibatch_index_iterator(num_training_examples, batch_size)
        for batch_idxs in batches:
            question_batch = question[batch_idxs]
            context_batch = context[batch_idxs]
            answer_batch = answer[batch_idxs]
            loss = model.predict(question_batch, context_batch, answer_batch)
            logging.info("loss: {}".format(loss))


def main(_):
    # Load and initialize the model
    model_class = load_class(tf.flags.FLAGS.model)
    if not issubclass(model_class, SquadModel):
        logging.fatal("Error! Given model {} is not an instance of core.SquadModel.".format(tf.flags.FLAGS.model))
        sys.exit(-1)

    # Load GloVe vectors
    glove_mat = np.load(tf.flags.FLAGS.embed_path)

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
    model = model_class(glove_mat)
    model.initialize_graph(config)

    mode = tf.flags.FLAGS.mode

    if mode == "train":
        train_model(model)
    elif mode == "eval":
        eval_model(model)


if __name__ == '__main__':
    # Setup arguments
    tf.flags.DEFINE_string("mode", "train", "Mode to run in, either \"train\" or \"eval\"")
    tf.flags.DEFINE_string("model", "models.baseline.BaselineModel", "full Python package path to a SquadModel")
    tf.flags.DEFINE_string("cell_type", "lstm", "type of RNN cell for training (either 'lstm' or 'gru'")

    # Training flags
    tf.flags.DEFINE_integer("epochs", 50, "number of epochs of training")
    tf.flags.DEFINE_integer("checkpoint_freq", 1, "epochs of training between re-evaluating and saving model")
    tf.flags.DEFINE_integer("max_length", 250, "maximum length of context passages (in tokens)")
    tf.flags.DEFINE_integer("hidden_size", 20, "size of the hidden state for the RNN cell")
    tf.flags.DEFINE_integer("embed_size", 100, "dimensionality of embedding")
    tf.flags.DEFINE_integer("batch_size", 30, "size of minibatches")

    tf.flags.DEFINE_float("keep_prob", 0.99, "inverse of drop probability")

    tf.flags.DEFINE_string("data_path", "data/squad/train.npz", "Path to .npz file holding the eval/training matrices")
    tf.flags.DEFINE_string("save_dir", "save", "path to save training results and model checkpoints")
    tf.flags.DEFINE_string("embed_path", "data/squad/glove.squad.100d.npy", "path to npy file holding trimmed embeddings")

    # Execute main() above, see tf.app documentation for details.
    tf.app.run()
