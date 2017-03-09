import sys
from importlib import import_module

import tensorflow as tf
import tqdm

from core import Config, SquadModel


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
            print("loading package {}".format(pkg))
            p = import_module(pkg)
            print("loading class {}".format(cls))
            return getattr(p, cls)
        else:
            print("direct loading {}".format(cls))
            return globals()[clsname]
    except AttributeError or KeyError:
        print("Error! Could not load class {}".format(clsname))
        sys.exit(-1)


def main(_):
    # Load and initialize the model
    model_class = load_class(tf.flags.FLAGS.model)
    if not issubclass(model_class, SquadModel):
        print("Error! Given model {} is not an instance of core.SquadModel.".format(tf.flags.FLAGS.model))
        sys.exit(-1)

    config = Config(dict(
        max_length=tf.flags.FLAGS.max_length,
        keep_prob=tf.flags.FLAGS.keep_prob,
        num_classes=2,
        embedding_size=tf.flags.FLAGS.embed_size,
        hidden_size=tf.flags.FLAGS.hidden_size,
        cell_type=tf.flags.FLAGS.cell_type,
    ))
    model = model_class()
    model.initialize(config)

    for epoch in tqdm.trange(tf.flags.FLAGS.epochs):
        if epoch % tf.flags.FLAGS.checkpoint_freq == 0:
            # Perform evaluation on the smaller dev set.
            model.checkpoint(tf.flags.FLAGS.save_dir)


if __name__ == '__main__':

    # Setup arguments
    tf.flags.DEFINE_string("model", "models.baseline.BaselineModel", "full Python package path to a SquadModel")
    tf.flags.DEFINE_string("cell_type", "lstm", "type of RNN cell for training (either 'lstm' or 'gru'")

    tf.flags.DEFINE_integer("epochs", 50, "number of epochs of training")
    tf.flags.DEFINE_integer("checkpoint_freq", 1, "epochs of training between re-evaluating and saving model")
    tf.flags.DEFINE_integer("max_length", 250, "Maximum length of sentences (in tokens)")
    tf.flags.DEFINE_integer("hidden_size", 150, "Size of the hidden state for the RNN cell")
    tf.flags.DEFINE_integer("embed_size", 100, "dimensionality of embedding")

    tf.flags.DEFINE_float("keep_prob", 0.99, "inverse of drop probability")

    tf.flags.DEFINE_string("save_dir", "save", "path to save training results and model checkpoints")
    tf.flags.DEFINE_string("embed_path", "data/squad/glove.trimmed.100.npz", "path to npz file holding word embeddings")

    tf.app.run()
