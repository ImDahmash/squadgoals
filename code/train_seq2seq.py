"""Dan Shiferaw's attempt at baseline seq2seq model
    Specifications:
        encoder:
        decoder:
        ...
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys

import json

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import ops


import numpy as np

from os.path import join as pjoin

from tqdm import tqdm

from evaluate import exact_match_score, f1_score


import logging


logging.basicConfig(level=logging.INFO)



####################################################################
# PARAMS
####################################################################



tf.app.flags.DEFINE_float("learning_rate", 0.01, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 5, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of model layers")
tf.app.flags.DEFINE_integer("vocab_dim", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")

####################################################################
# NEW PARAMS
####################################################################

# sequence length for both encoder and decoder
tf.app.flags.DEFINE_string("seq_length", 781, "Sequence length for encoder/decoder")

# train files
tf.app.flags.DEFINE_string("train_question_ids_path", "train.ids.question", "Path to the train ids question file (default: ./data/squad/train.ids.question)")
tf.app.flags.DEFINE_string("train_context_ids_path", "train.ids.context", "Path to the train ids context file (default: ./data/squad/train.ids.context)")
tf.app.flags.DEFINE_string("train_answer_path", "train.span", "Path to the train answer start and end indices file (default: ./data/squad/train.span)")


# val files
tf.app.flags.DEFINE_string("val_question_ids_path", "val.ids.question", "Path to the val ids question file (default: ./data/squad/val.ids.question)")
tf.app.flags.DEFINE_string("val_context_ids_path", "val.ids.context", "Path to the val ids context file (default: ./data/squad/val.ids.context)")
tf.app.flags.DEFINE_string("val_answer_path", "val.span", "Path to the val answer start and end indices file (default: ./data/squad/val.span)")

# compressed train and val data
tf.app.flags.DEFINE_string("train_data_path", "./data/squad/train_data.npz", "Path to train data")
tf.app.flags.DEFINE_string("val_data_path", "./data/squad/val_data.npz", "Path to val data")




FLAGS = tf.app.flags.FLAGS

_PAD = b"<pad>"
_SOQ = b"<soq>"
_UNK = b"<unk>"

_START_VOCAB = [_PAD, _SOQ, _UNK]

PAD_ID = 0
SOQ_ID = 1
UNK_ID = 2





class Seq2Seq(object):
    def __init__(self,  pretrained_embeddings):
        self.seq_length = FLAGS.seq_length
        self.pretrained_embeddings = pretrained_embeddings
        self.num_layers = FLAGS.num_layers
        self.state = None
        self.embeddings_dim = None
        self.embeddings = None
        self.batch_size = FLAGS.batch_size
        self.num_classes = 2 # in answer or not in answer
        self.enc_input_placeholder = None
        self.dec_output_placeholder = None

        self.build()


    def add_placeholders(self):
        """
        Adds following nodes to the encoder computational graph

        input_placeholder: Input placeholder tensor of  shape (None, self.input_seq_length), type tf.float32
        mask_placeholder:  Mask placeholder tensor of shape (None, self.input_seq_length), type tf.bool
        """
        # weird: we may need 1 for third dim...
        self.enc_input_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, self.seq_length, 1), name="enc_input_placeholder")
        self.dec_output_placeholder = tf.placeholder(tf.float64, shape=(self.batch_size, self.seq_length), name="dec_output_placeholder")


    def create_feed_dict(self, inputs_batch, outputs_batch=None):
        """Creates the feed_dict for the encoder.

        Args:
            enc_input_batch: A batch of input encoder data.
            mask_batch:   A batch of mask data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        feed_dict = dict()
        feed_dict[self.enc_input_placeholder] = inputs_batch
        if outputs_batch is not None:
            feed_dict[self.dec_output_placeholder] = outputs_batch
        return feed_dict

    def add_embedding(self):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
       """
        embeddings = tf.Variable(self.pretrained_embeddings)
        embeddings = tf.nn.embedding_lookup(embeddings, self.enc_input_placeholder)
        self.embeddings_dim = FLAGS.vocab_dim
        self.embeddings = tf.reshape(embeddings, [self.batch_size, self.seq_length, self.embeddings_dim])


    def encode(self):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial
        hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # so one column per input element in seq>

        # make sure to set scope for your rnn cells to prevent conflicts!
        with tf.variable_scope('encoder'):
            encode_cell = tf.contrib.rnn.LSTMCell(2, state_is_tuple=True)
            encode_cell = tf.contrib.rnn.MultiRNNCell([encode_cell] * self.num_layers, state_is_tuple=True)
            _, enc_state = tf.nn.dynamic_rnn(encode_cell, self.embeddings, dtype=tf.float64, \
                initial_state= self.state)
        self.state = enc_state
        return

    def decode(self):
        with tf.variable_scope('decoder'):
            # two-class problem
            decode_cell = tf.contrib.rnn.LSTMCell(2, state_is_tuple=True)
            decode_cell = tf.contrib.rnn.MultiRNNCell([decode_cell] * self.num_layers, state_is_tuple=True)
            initial_state = self.state
            outputs, state = tf.nn.dynamic_rnn(decode_cell, self.embeddings, dtype=tf.float64, \
                initial_state=initial_state)

        
        
        return outputs


    def add_loss_op(self, pred):
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, \
            labels=tf.to_int32(self.dec_output_placeholder)))
        return loss

    def add_train_op(self, loss):
        train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(loss)
        return train_op

    def build(self):
        self.add_placeholders()
        self.add_embedding()
        self.encode()
        self.pred = self.decode()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_train_op(self.loss)



def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model


def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)


def read_dataset(train_question_path, train_context_path, train_answer_path):
    """Reads the dataset, extracts context, question, answer spans,
    and answers."""

    train_question_filename = train_question_path
    train_context_filename = train_context_path
    train_answer_filename = train_answer_path

    input_data = []
    answer_labels = [] # which tokens in context part of answer or not
    answer_data = []



    with open(train_question_filename) as question_file,  \
         open(train_context_filename) as context_file,\
         open(train_answer_filename) as answer_file:
         question_data = [[1] + [int(x) for x in line.strip('\n').split()] \
            for line in question_file]
         input_data = [[int(x) for x in line.strip('\n').split()] + question_data[i]\
            for i,line in enumerate(context_file)]       
         #FLAGS.seq_length = max([len(x) for x in input_data])
         answer_spans = np.array([np.array([int(x) for x in line.strip('\n').split()]) \
            for line in answer_file])
         answer_data = np.array([context[answer_spans[i][0]:answer_spans[i][1] + 1] \
            for i, context in enumerate(input_data)])
         # zero padding
         input_data = np.array([[x[i] if i < len(x) else 0 for i in xrange(FLAGS.seq_length)]    for x in input_data])
         answer_data = np.array([[1 if (i < len(x) and i >= answer_spans[i][0] and i <= answer_spans[i][1]) \
            else 0 for i in xrange(FLAGS.seq_length)]  for x in input_data])

    return input_data, answer_spans, answer_data





def round_down(num, divisor):
    return num - (num%divisor)    



def gen_batches(epochs, batch_size, train_data):
    train_data_len = len(train_data)
    for i in range(epochs):
        np.random.shuffle(train_data)
        for ndx in tqdm(range(0, round_down(int(train_data_len / 10),10), batch_size)):
            yield (train_data[ndx:ndx+batch_size], ndx, ndx + batch_size) 


 
   
def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """

    answers = {}
    feed = model.create_feed_dict(inputs_batch)
    predictions = sess.run(model.pred, feed_dict=feed)

    return answers


def train(model, train_data, val_data):
    tf.set_random_seed(2000)
    saver = tf.train.Saver()
    train_dir = FLAGS.train_dir
    with tf.Session() as sess:

        """
        ckpt = tf.train.get_checkpoint_state(train_dir)
        v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
        if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
            logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(session, ckpt.model_checkpoint_path)
         lse:
            logging.info("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())
            logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        """
        logging.info("Created model with fresh parameters.")
        sess.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        training_losses = []
        val_losses = []
        min_val = float("inf")
       # num_batches = len(train_data) / FLAGS.batch_size
        steps = 0
        avg_train_loss = 0
        avg_val_loss = 0
        for idx, batch_tuple in enumerate(gen_batches(FLAGS.epochs, FLAGS.batch_size, train_data)):
            train_batch = batch_tuple[0]
            val_batch = val_data[batch_tuple[1]: batch_tuple[2]]
            do_val = True if len(val_batch) == 30 else False
            steps += 1
            

            train_feed_dict = model.create_feed_dict(np.expand_dims(np.array([item[0] for item in train_batch],), axis=2), \
                np.array([item[1] for item in train_batch],))
            _, train_loss = sess.run([model.train_op, model.loss], feed_dict=train_feed_dict)
            training_losses.append(avg_train_loss)
            avg_train_loss += train_loss
            
            if do_val:
                val_feed_dict = model.create_feed_dict(np.expand_dims(np.array([item[0] for item in val_batch],), axis=2), \
                    np.array([item[1] for item in val_batch],))
                val_loss = model.loss.eval(feed_dict=val_feed_dict)
                avg_val_loss += val_loss


            

            if steps == 10:
                steps = 0
                print("Training loss for previous 10 batches", ":", avg_train_loss / 10)
                training_losses.append(avg_train_loss)
                avg_train_loss = 0

                if do_val:
                    val_losses.append(avg_val_loss)
                    print("Val loss for previous 10 batches", ":", avg_val_loss / 10)
                    if val_loss < min_val:
                        min_val = val_loss
                        print("New min val loss!")
                        saver.save(sess, "train/model.ckpt." + str(idx))
                avg_val_loss = 0



    return training_losses



""" Handle overall training process for seq2Seq baseline model
"""
def main(_):

    # load embeddings 
    embed_path = FLAGS.embed_path or pjoin("data", "squad", \
        "glove.trimmed.{}.npz".format(FLAGS.vocab_dim))
    embed_vals = np.load(embed_path)
    pretrained_embeddings = embed_vals['glove']

   

    """
    # download and save training data for later
    train_input_data, train_answer_spans, train_answer_data = \
        read_dataset(pjoin(FLAGS.data_dir, FLAGS.train_question_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.train_context_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.train_answer_path))
    train_data_path = FLAGS.train_data_path
    np.savez_compressed(train_data_path, input_data=train_input_data, answer_spans=train_answer_spans, \
        answer_data = train_answer_data)
    # download and save val data for later
    val_input_data, val_answer_spans, val_answer_data = \
        read_dataset(pjoin(FLAGS.data_dir, FLAGS.val_question_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.val_context_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.val_answer_path))
    val_data_path = FLAGS.val_data_path
    np.savez_compressed(val_data_path, input_data=val_input_data, answer_spans=val_answer_spans, \
        answer_data = val_answer_data)
    """
    # load train data
    train_data_path = FLAGS.train_data_path
    loaded_train_data = np.load(train_data_path)
    train_input_data = loaded_train_data['input_data']
    train_answer_spans = loaded_train_data['answer_spans']
    train_answer_data = loaded_train_data['answer_data']


    # load val data
    val_data_path = FLAGS.val_data_path
    loaded_val_data = np.load(val_data_path)
    val_input_data = loaded_val_data['input_data']
    val_answer_spans = loaded_val_data['answer_spans']
    val_answer_data = loaded_val_data['answer_data']


    # prepare vocabulary for both train and val
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)


    train_data = zip(train_input_data, train_answer_data)
    val_data = zip(val_input_data, val_answer_data)



     # initialize model
    seq2Seq= Seq2Seq(pretrained_embeddings=pretrained_embeddings)

    train(seq2Seq, train_data, val_data)

    

   # answers = generate_answers(sess, seq2Seq, , rev_vocab)

    # save model?

    # write to json file to root dir
   # with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
       # f.write(unicode(json.dumps(answers, ensure_ascii=False)))

    



if __name__ == "__main__":
    tf.app.run()