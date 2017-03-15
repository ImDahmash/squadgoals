from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


import math
import os
import time
from pprint import pprint

import numpy as np
import tensorflow as tf
# from tensorflow.contrib.rnn import DropoutWrapper, BasicLSTMCell, GRUCell, MultiRNNCell, RNNCell

from models import MatchLSTMModel, BiLSTMModel
from utils import minibatch_index_iterator, Progress


# tfdbg tensorflow debugger
# from tensorflow.python import debug as tf_debug


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
tf.flags.DEFINE_integer('batch_size', 30, 'size of mini-batches')
tf.flags.DEFINE_integer('embed_dim', 100, 'embedding dimension')
tf.flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
tf.flags.DEFINE_integer('layers', 2, 'number of hidden layers')

tf.flags.DEFINE_string('model', 'match', 'type of model, either "match" for Match-LSTM w/Answer Pointer or "bilstm" for simple BiLSTM baseline.')
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


###############################################################
#           Training
###############################################################


def main(_):
    with tf.Session().as_default() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)   # DEbugging

        # Train the model
        config = tf.flags.FLAGS
        print("Configuration:")
        pprint(tf.flags.FLAGS.__dict__['__flags'], indent=4)

        # Time building the graph
        tic = time.time()
        if tf.flags.FLAGS.model == "match":
            model = MatchLSTMModel(config).build_graph()
        else:
            model = BiLSTMModel(config).build_graph()
        toc = time.time()
        print("Took {:.2f}s to build graph.".format(toc - tic))

        sess.run(tf.global_variables_initializer())

        print("Parameters:")
        pprint(list(kv.name for kv in tf.trainable_variables()))
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
            bar = Progress('Epoch {} of {}'.format(epoch + 1, config.epochs), steps=num_batches, width=20, sameline=False)
            for batch, idxs in enumerate(minibatch_index_iterator(num_examples, config.batch_size)):
                # Read batch_size indexes for constructing training batch
                qs = questions[idxs]
                cs = contexts[idxs]
                ans = answers[idxs]
                q_ls = q_lens[idxs]
                c_ls = c_lens[idxs]

                # Perform train step
                loss = model.train(qs, cs, ans, q_ls, c_ls)
                print("loss", loss)
                losses.append(loss)

                # Calculate some stats to print
                bar.tick(loss=loss, avg=np.average(losses), hi=max(losses), lo=min(losses))
                saver.save(sess, save_path, global_step=epoch)
            avg_loss = np.average(losses)
            epoch_losses.append(avg_loss)
            print("\n--- Epoch {} Average Train Loss: {:.7f}".format(epoch + 1, avg_loss))
            # Run validation, get validation loss
            # val_loss = model.evaluate(val_qs, val_cs, val_as, val_q_lens, val_c_lens)
            # print("  \ Validation Loss: {:.7f}".format(val_loss))
            # Save the model
            # saver.save(sess, save_path, global_step=epoch)

        # Write the losses out to a file for later
        print("Saving statistics...")
        np.save("statistics.npz", epoch_losses=epoch_losses)


if __name__ == '__main__':
    tf.app.run(main=main)
