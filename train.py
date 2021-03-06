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

from models import MatchLSTMModel
from utils import minibatch_index_iterator, Progress


###############################################################
#       Code to setup the environment, configuration, etc.
###############################################################


"""
Configuration options:
"""
tf.flags.DEFINE_integer('hidden_size', 150, "Size of hidden states for encoder")
tf.flags.DEFINE_integer('batch_size', 30, 'size of mini-batches')
tf.flags.DEFINE_integer('embed_dim', 300, 'embedding dimension')
tf.flags.DEFINE_integer('epochs', 10, 'number of epochs for training')
tf.flags.DEFINE_integer('layers', 2, 'number of hidden layers')

tf.flags.DEFINE_string('cell_type', 'lstm', "Cell type for RNN")
tf.flags.DEFINE_float('lr', 0.01, 'learning rate')

tf.flags.DEFINE_integer('subset', 0, 'If > 0, only trains on a subset of the train data of given size')

tf.flags.DEFINE_float('keep_prob', 0.6, 'Keep probability for dropout.')

tf.flags.DEFINE_string('embed_path', 'data/squad/glove.squad.300d.npy', "Path to a .npy file holding the GloVe vectors")
tf.flags.DEFINE_string('train_path', 'data/squad/train.npz', "Path to training data as an .npz file")
tf.flags.DEFINE_string('val_path', 'data/squad/val.npz', "Path to validation data as an .npz file")
tf.flags.DEFINE_string('save_dir', 'save', 'directory to save model checkpoints after each epoch')

tf.flags.DEFINE_boolean('resume', False, 'Resume from latest checkpoint in save_dir when flags is passed. Default is not to')
tf.flags.DEFINE_boolean('save', True, 'Checkpoint the model after each epoch.')
tf.flags.DEFINE_boolean('valid', True, 'Perform validation periodically.')

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
    print("Starting")
    with tf.Session().as_default() as sess:
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)   # DEbugging

        # Train the model
        config = tf.flags.FLAGS
        print("Configuration:")
        pprint(tf.flags.FLAGS.__dict__['__flags'], indent=4)

        # Time building the graph
        tic = time.time()
        model = MatchLSTMModel(config).build_graph()
        toc = time.time()
        print("Took {:.2f}s to build graph.".format(toc - tic))

        sess.run(tf.global_variables_initializer())
        print("Initialized globals.")

        print("Parameters:")
        pprint(list(kv.name for kv in tf.trainable_variables()))
        questions, contexts, answers, q_lens, c_lens = load_data(config.train_path)

        num_examples = config.subset if config.subset > 0 else questions.shape[0]
        total_params = sum(v.get_shape().num_elements() for v in tf.trainable_variables())
        print("Total params: {}".format(total_params))


        # Setup saving
        saver = tf.train.Saver(max_to_keep=0)  # Don't delete checkpoint files, cleanup manually
        save_path = os.path.join(config.save_dir, "model")
        best_path = os.path.join(config.save_dir, "model_best")
        if not tf.gfile.Exists(config.save_dir):
            tf.gfile.MakeDirs(config.save_dir)

        # If we want to restore then try it
        # Resume training
        if tf.flags.FLAGS.resume:
            latest = tf.train.latest_checkpoint(config.save_dir)
            print("Restoring from {}  ...".format(latest))
            saver.restore(sess, latest)

        # Perform training pass
        all_losses = []
        epoch_losses = []
        validation_losses = []
        print("Begin training!")

        for epoch in range(config.epochs):
            num_batches = math.ceil(num_examples / config.batch_size)
            losses = []

            # Halve the learning rate after each epoch
            lr = config.lr * (0.5)**epoch

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
                loss, norm = model.train(qs, cs, ans, q_ls, c_ls, norms=True, lr=lr)
                losses.append(loss)
                all_losses.append(loss)

                # Calculate some stats to print
                bar.tick(loss=loss, avg10=np.average(losses[-10:]), hi=max(losses), lo=min(losses), norm=float(norm), lr=lr)

                if tf.flags.FLAGS.save and (batch % 100 == 0):
                    print("Checkpointing...")
                    saver.save(sess, save_path, global_step=epoch)
                    print("Done")

                # Calculate validation
                if config.valid and batch % 500 == 0 and batch > 0:
                    # Perform validation here
                    val_loss = validation(sess, config, model)
                    if len(validation_losses) == 0 or val_loss < np.min(validation_losses):
                        # Save best model
                        print("===> New best validation! Saving to {}".format(best_path))
                        saver.save(sess, best_path)

                    validation_losses.append(val_loss)
                    # Write the losses out to a file for later
                    print("Saving statistics...")
                    np.savez("statistics.npz", epoch_losses=epoch_losses, validation=validation_losses)


            if config.valid:
                val_loss = validation(sess, config, model)
                validation_losses.append(val_loss)
                if len(validation_losses) == 0 or val_loss < np.min(validation_losses):
                    # Save best model
                    print("===> New best validation! Saving to {}".format(best_path))
                    saver.save(sess, best_path)


            avg_loss = np.average(losses)
            epoch_losses.append(avg_loss)
            print("\n--- Epoch {} Average Train Loss: {:.7f}".format(epoch + 1, avg_loss))

        # Write the losses out to a file for later
        if config.save:
            print("Saving statistics...")
            np.savez("statistics.npz", epoch_losses=epoch_losses, validation=validation_losses)


def validation(sess, config, model):
    print("Performing validation...")
    val_qs, val_cs, val_as, val_q_lens, val_c_lens = load_data(config.val_path)
    num_examples = val_qs.shape[0]
    BATCH_SIZE = 30
    losses = []
    for i in range(0, num_examples, BATCH_SIZE):
        qs = val_qs[i:i+BATCH_SIZE]
        cs = val_cs[i:i+BATCH_SIZE]
        ans = val_as[i:i+BATCH_SIZE]
        qlens = val_q_lens[i:i+BATCH_SIZE]
        clens = val_c_lens[i:i+BATCH_SIZE]
        loss = model.evaluate(qs, cs, ans, qlens, clens)
        losses.append(loss)
    val_loss = np.average(losses)
    print("  \ Validation Loss: {:.7f}".format(val_loss))
    return val_loss


if __name__ == '__main__':
    tf.app.run(main=main)
