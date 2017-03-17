"""baseline dynamic coattention model of sorts
   
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import sys
import re

import json

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile




import numpy as np

from os.path import join as pjoin

from tqdm import tqdm

from evaluate import exact_match_score, f1_score


from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map


import logging
import io


logging.basicConfig(level=logging.INFO)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 
PARAMS
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

tf.app.flags.DEFINE_float("learning_rate", .0001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", .8, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 16, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 10, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 256, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of model layers")
tf.app.flags.DEFINE_integer("embed_dim", 300, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("train_dir", "train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{embed_dim}.npz)")
tf.app.flags.DEFINE_string("dev_path", "data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")


# sequence length for both encoder and decoder
tf.app.flags.DEFINE_string("question_seq_length", 20, "Input sequence length for encoder/decoder")
tf.app.flags.DEFINE_string("context_seq_length", 200, "Input sequence length for encoder/decoder")


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
tf.app.flags.DEFINE_string("dev_data_path", "./data/squad/dev_data.npz", "Path to dev data (just input and question uuid data)")

# embeddings and summaries
tf.app.flags.DEFINE_string("glove_dir", "data/dwr", "Path to the trimmed GLoVe embedding (default: data/dwr")
tf.app.flags.DEFINE_string("summaries_dir", "log", "Path to the summaries from Tensorboard")



FLAGS = tf.app.flags.FLAGS
_PAD = b"<pad>"
_SOA = b"<soa>"
_EOA = b"<eoa>"
_UNK = b"<unk>"


_START_VOCAB = [_PAD, _SOA, _EOA, _UNK]

PAD_ID = 0 # padding
SOA_ID = 1 # start of answer/decoding
EOA_ID = 2 # end of answer/decoding
UNK_ID = 3 # unknown token

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 
Further data downloading and processing for seq2seq. Had to remain within this file b/c of issues 
with defining and importing FLAGS
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def read_train_dataset(train_question_path, train_context_path, train_answer_path):
    """Reads the train/val (sorry for bad naming) dataset, extracts question,  context, decoder input,
    and decoder output data."""

    train_question_filename = train_question_path
    train_context_filename = train_context_path
    train_answer_filename = train_answer_path

    question_data = []
    context_data = []
    decoder_input_data = []
    decoder_output_data = []





    with open(train_question_filename) as question_file,  \
         open(train_context_filename) as context_file,\
         open(train_answer_filename) as answer_file:
        question_data = [[int(x) for x in line.strip('\n').split()] \
            for line in question_file]
        context_data  = [[int(x) for x in line.strip('\n').split()]\
            for i,line in enumerate(context_file)]
        answer_spans = np.array([np.array([int(x) for x in line.strip('\n').split()]) \
            for line in answer_file])    
        # zero padding
        question_data = np.array([[x[i] if i < len(x) else 0 for i in xrange(FLAGS.question_seq_length)] \
            for x in question_data])
        context_data = np.array([[x[i] if i < len(x) else 0 for i in xrange(FLAGS.context_seq_length)] \
            for x in context_data])

    

        

    return question_data, context_data, answer_spans



def read_dev_dataset(dev_dataset, tier, vocab):
    """Reads the dev dataset json file and extracts the input data (context and question
    vectors) and question uuid data.
    """

    dev_question_data = []
    dev_context_data = []
    dev_question_uuid_data = []

    for articles_id in tqdm(range(len(dev_dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dev_dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [int(vocab.get(w, UNK_ID)) for w in context_tokens]
                question_ids = [int(vocab.get(w, UNK_ID)) for w in question_tokens]

            
           
                dev_question_datum = np.array([question_ids[i] if i < len(question_ids) \
                    else 0 for i in xrange(FLAGS.question_seq_length)])
                dev_question_data.append(dev_question_datum)

                dev_context_datum = np.array([context_ids[i] if i < len(context_ids) \
                    else 0 for i in xrange(FLAGS.context_seq_length)])
                dev_context_data.append(dev_context_datum)

                dev_question_uuid_data.append(question_uuid)

    return dev_question_data, dev_context_data, dev_question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    print("Downloading {}".format(dev_filename))
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    dev_question_data, dev_context_data, dev_question_uuid_data = read_dev_dataset(dev_data, 'dev', vocab)

    return dev_question_data, dev_context_data, dev_question_uuid_data

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 
Highway Networks and also Tensorboard summary data
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries' + '_' + name):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def weight_bias(name,W_shape, b_shape):
    W = tf.get_variable(name=name + "_weight",dtype=tf.float64, \
        shape=[W_shape, b_shape],initializer=tf.contrib.layers.xavier_initializer()) 
    #variable_summaries(W)
    b = tf.get_variable(name=name + "_bias",shape=[b_shape], initializer=tf.constant_initializer(0.0), \
                dtype=tf.float64)
  #  variable_summaries(b)
    return W, b

def dense_layer(x, W_shape, b_shape, activation):
    W, b = weight_bias("dense",W_shape, b_shape)
    pre_activation = tf.matmul(x, W) + b
  #  tf.summary.histogram('highway_dense_pre_activations',  pre_activation)
    activation = activation(pre_activation)
    activation = tf.nn.dropout(activation,keep_prob=FLAGS.dropout)
  #  tf.summary.histogram('highway_dense_activations',  activation)
    return activation


def output_layer(x, W_shape, b_shape):
    W, b = weight_bias("output",W_shape, b_shape)
    output = tf.matmul(x, W) + b
    output = tf.nn.dropout(output,keep_prob=FLAGS.dropout)
  #  tf.summary.histogram('highway_output',  output)
    return output

def highway_layer(x, size, activation, carry_bias=-1.0):
    W, b = weight_bias("first",size, size)

    with tf.name_scope('transform_gate'):
        W_T, b_T = weight_bias("second",size, size)

    H_pre_activation = tf.matmul(x, W) + b
  #  tf.summary.histogram('h_pre_activation',   h_pre_activation)

    H = activation(H_pre_activation , name='activation')
  #  tf.summary.histogram('H',   H)

    T_pre_activation = tf.matmul(x, W_T) + b_T
  #  tf.summary.histogram('T_pre_activation',   T_pre_activation)

    T = tf.sigmoid(T_pre_activation, name='transform_gate')
 #   tf.summary.histogram('T',   T)

    C = tf.subtract(tf.cast(1.0, tf.float64), T, name="carry_gate")
  #  tf.summary.histogram('C',   C)

    H_T = tf.multiply(H, T)
 #   tf.summary.histogram('H_T',   H_T)

    x_C =  tf.multiply(x, C)
   # tf.summary.histogram('x_C',   x_C)
    
    y = tf.add(H_T, x_C, name='y') # y = (H * T) + (x * C)
  #  tf.summary.histogram('y',   y)
    return y


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 
seq2seq model
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class Seq2Seq(object):
    def __init__(self,  pretrained_embeddings):
        self.pretrained_embeddings = pretrained_embeddings
        self.build()


    def add_placeholders(self):
        """
        """
        self.question_input = tf.placeholder(tf.int32, shape=(None, FLAGS.question_seq_length), name="question_input")
        self.question_seq_lens = tf.placeholder(tf.int32, shape=(None, ), name="question_seq_lens")
        self.context_input = tf.placeholder(tf.int32, shape=(None, FLAGS.context_seq_length), name="context_input")
        self.context_seq_lens = tf.placeholder(tf.int32, shape=(None, ), name="context_seq_lens")
        self.mask = tf.placeholder(tf.float64, shape=(None, FLAGS.context_seq_length, 1), name="mask")
        self.start_decoder_labels= tf.placeholder(tf.int32, shape=(None,), name="start_decoder_labels")
        self.end_decoder_labels= tf.placeholder(tf.int32, shape=(None, ), name="end_decoder_labels")


    def create_feed_dict(self, question_inputs, question_seq_lens, context_inputs, \
        context_seq_lens, start_decoder_labels=None, end_decoder_labels=None):
        """Creates the feed_dict for the encoder.
        """
        batch_size = question_inputs.shape[0]

        feed_dict = dict()
        feed_dict[self.question_input] = question_inputs
        feed_dict[self.question_seq_lens] = question_seq_lens
        feed_dict[self.context_input] = context_inputs
        feed_dict[self.context_seq_lens] = context_seq_lens

        mask = np.zeros([batch_size, FLAGS.context_seq_length, 1])
        for i in range(batch_size):
            end = context_seq_lens[i]
            mask[i, end:] = -1000.0

        feed_dict[self.mask] = mask
        if start_decoder_labels is not None:
            feed_dict[self.start_decoder_labels] = start_decoder_labels
        if end_decoder_labels is not None:
            feed_dict[self.end_decoder_labels] = end_decoder_labels
       
        return feed_dict

    def add_embedding(self, input_tokens, seq_length):
        """Adds an embedding layer that maps from input tokens (integers) to vectors and then
        concatenates those vectors:
       """
        embeddings = tf.constant(self.pretrained_embeddings,name="embeddings")
        embeddings = tf.nn.embedding_lookup(embeddings, input_tokens)
        return embeddings


    def encode(self):
        """
        """
     
        batch_size = tf.shape(self.context_input)[0]
      #  encode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
      #          initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
        encode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
                 state_is_tuple=True)
        encode_cell = tf.contrib.rnn.DropoutWrapper(encode_cell, output_keep_prob=FLAGS.dropout)
        stacked_encode_cell = tf.contrib.rnn.MultiRNNCell([encode_cell] * FLAGS.num_layers, state_is_tuple=True)

        with tf.variable_scope('context_encoder') as scope:
            context_inputs = self.add_embedding(self.context_input, FLAGS.context_seq_length)
            self.context_inputs = context_inputs
            
            context_outputs, context_state = tf.nn.dynamic_rnn(stacked_encode_cell, \
                context_inputs, dtype=tf.float64, sequence_length=self.context_seq_lens)
            self.context_outputs = context_outputs
            
        with tf.variable_scope('question_encoder') as scope:
            question_inputs = self.add_embedding(self.question_input, FLAGS.question_seq_length)
            self.question_inputs = question_inputs

            question_outputs, question_state = tf.nn.dynamic_rnn(stacked_encode_cell, \
                question_inputs, dtype=tf.float64, sequence_length=self.question_seq_lens)
            self.question_outputs = tf.identity(question_outputs)

        b_q = tf.get_variable("b_q", [FLAGS.state_size], initializer=tf.constant_initializer(0.0), \
                dtype=tf.float64)
        U_q = tf.get_variable(name="U_q",dtype=tf.float64,
                    shape=[FLAGS.state_size, FLAGS.state_size],
                    initializer=tf.contrib.layers.xavier_initializer())    
        question_outputs = tf.reshape(question_outputs, [-1, FLAGS.state_size])
        # introduce non-linear projection layer on top of question encoding to allow for variation b/t
        # question encoding space and document encoding space
        question_outputs = tf.tanh((tf.matmul(question_outputs, U_q) + b_q))
        question_outputs = tf.reshape(question_outputs, [-1, FLAGS.question_seq_length, FLAGS.state_size])
        self.proj_question_outputs = question_outputs
      #  self.question_outputs = question_outputs



        
        # now implement coattention mechanism
        with tf.variable_scope('coattention_encoder'):
            # transpose for affinity matrix multiplication later
            affinity = tf.matmul(context_outputs, tf.transpose(question_outputs, perm=[0, 2, 1]))
            # now we get attention weights for context across each word in question
            affinity_question = tf.nn.softmax(affinity)
            # now we get attention weight for question across each word in context
            affinity_context = tf.nn.softmax(tf.transpose(affinity, perm=[0, 2, 1]))
            # compute the summaries, or attention contexts, of the document in light of each word of the question
            attention_question= tf.matmul(tf.transpose(context_outputs, perm=[0, 2, 1]), affinity_question)
            # finally we compute codependent representation of question and document
            coattention_context = tf.matmul(tf.concat([tf.transpose(question_outputs, perm=[0, 2, 1]), attention_question], 1), affinity_context)
            # transpose to prepare for decoding
            decoder_input = tf.concat([tf.transpose(coattention_context, perm=[0, 2, 1]), context_outputs], 2)
            self.decoder_input = decoder_input
            """
            fw_co_encode_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
                initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
            bw_co_encode_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size,  use_peepholes=True,\
                initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
            """

            fw_co_encode_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
                 state_is_tuple=True)
            bw_co_encode_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size,  use_peepholes=True,\
                state_is_tuple=True)

            fw_co_encode_decode_cell = tf.contrib.rnn.DropoutWrapper(fw_co_encode_decode_cell, output_keep_prob=FLAGS.dropout)
            bw_co_encode_decode_cell = tf.contrib.rnn.DropoutWrapper(bw_co_encode_decode_cell, output_keep_prob=FLAGS.dropout)
            co_encode_decode_outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_co_encode_decode_cell, \
                bw_co_encode_decode_cell, decoder_input, sequence_length=self.context_seq_lens, dtype=tf.float64)
            co_encode_decode_outputs = tf.concat(co_encode_decode_outputs, 2)
            self.co_encode_decode_outputs = co_encode_decode_outputs

        

        return co_encode_decode_outputs



    def decode(self, decoder_input):

 
        

        batch_size = tf.shape(self.context_input)[0]
        input_layer_size = FLAGS.state_size 
        hidden_layer_size = FLAGS.state_size # use ~71 for fully-connected (plain) layers, 50 for highway layers
        output_layer_size = 1
        layer_count = 2
        carry_bias_init = -1.0

        # predict start of span
        with tf.variable_scope('start_decoder'):
         #   start_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
         #       initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
            start_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size, use_peepholes=True, \
             state_is_tuple=True)
            start_decode_cell = tf.contrib.rnn.DropoutWrapper(start_decode_cell, output_keep_prob=FLAGS.dropout)
            stacked_start_decode_cell = tf.contrib.rnn.MultiRNNCell([start_decode_cell] * FLAGS.num_layers, state_is_tuple=True)
            start_decode_outputs, start_decode_state = tf.nn.dynamic_rnn(stacked_start_decode_cell, \
                decoder_input, sequence_length=self.context_seq_lens, dtype=tf.float64)
           # start_decode_state = start_decode_state[FLAGS.num_layers- 1]
            # prepare to work on each batch
            """
            start_decode_outputs = tf.transpose(start_decode_outputs, [1, 0, 2])
            start_decode_outputs = tf.gather_nd(start_decode_outputs, [[self.context_seq_lens[i] - 1,i] \
                for i in xrange(FLAGS.batch_size)])
            """
            start_decode_outputs = tf.reshape(start_decode_outputs, [-1, FLAGS.state_size])
            self.start_decode_outputs = start_decode_outputs
           

            prev_start_y = None
            start_logits = None
            for i in range(layer_count):
                with tf.variable_scope("layer{0}".format(i)) as scope:
                    if i == 0: # first, input layer
                        prev_start_y = dense_layer(start_decode_outputs, input_layer_size, hidden_layer_size, tf.nn.relu)
                    elif i == layer_count - 1: # last, output layer
                        start_logits = output_layer(prev_start_y, hidden_layer_size, output_layer_size)
                    else: # hidden layers
                        prev_start_y = highway_layer(prev_start_y, hidden_layer_size, tf.nn.relu, carry_bias=carry_bias_init)
            start_logits = tf.reshape(start_logits, [-1, FLAGS.context_seq_length, 1])
            start_logits = tf.squeeze(start_logits + self.mask)



          
        # predict end of span. conditional on current state of start decoder
        with tf.variable_scope('end_decoder'):
           # end_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size,  use_peepholes=True,\
         #       initializer=tf.contrib.layers.xavier_initializer(), state_is_tuple=True)
            end_decode_cell = tf.contrib.rnn.LSTMCell(FLAGS.state_size,  use_peepholes=True,\
                 state_is_tuple=True)
            end_decode_cell = tf.contrib.rnn.DropoutWrapper(end_decode_cell, output_keep_prob=FLAGS.dropout)
            stacked_end_decode_cell = tf.contrib.rnn.MultiRNNCell([end_decode_cell] * FLAGS.num_layers, state_is_tuple=True)
            end_decode_outputs, _ = tf.nn.dynamic_rnn(stacked_end_decode_cell, \
                decoder_input, sequence_length=self.context_seq_lens, dtype=tf.float64, initial_state=start_decode_state)
            """
            end_decode_outputs = tf.transpose(end_decode_outputs, [1, 0, 2])
            end_decode_outputs = tf.gather_nd(end_decode_outputs, [[self.context_seq_lens[i] - 1,i] \
                for i in xrange(FLAGS.batch_size)])
            """
            end_decode_outputs = tf.reshape(end_decode_outputs, [-1, FLAGS.state_size])
            self.end_decode_outputs = end_decode_outputs

            

            prev_end_y = None
            end_logits = None
            for i in range(layer_count):
                with tf.variable_scope("layer{0}".format(i)) as scope:
                    if i == 0: # first, input layer
                        prev_end_y = dense_layer(end_decode_outputs, input_layer_size, hidden_layer_size, tf.nn.relu)
                    elif i == layer_count - 1: # last, output layer
                        end_logits = output_layer(prev_end_y, hidden_layer_size, output_layer_size)
                    else: # hidden layers
                        prev_end_y = highway_layer(prev_end_y, hidden_layer_size, tf.nn.relu, carry_bias=carry_bias_init)


          
            end_logits = tf.reshape(end_logits, [-1, FLAGS.context_seq_length, 1])
            end_logits = tf.squeeze(end_logits + self.mask)

     
      
        return start_logits, end_logits


    def add_loss_op(self, start_pred, end_pred):
        start_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=start_pred, labels=self.start_decoder_labels)
        end_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=end_pred, labels=self.end_decoder_labels)
        variable_summaries(start_losses, "start_losses")
        variable_summaries(end_losses, "end_losses")



        start_losses = tf.reduce_mean(start_losses)
        end_losses = tf.reduce_mean(end_losses)
        total_loss = tf.reduce_sum([start_losses,end_losses])
        tf.summary.scalar("total_loss",total_loss)

        return total_loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        gradients = [gv[0] for gv in grads_and_vars]
        variables = [gv[1] for gv in grads_and_vars]

        gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.max_gradient_norm)

        grads_and_vars = zip(gradients, variables)
        train_op = optimizer.apply_gradients(grads_and_vars)
        grad = tf.global_norm(gradients) # to check learning
        tf.summary.scalar("grad",grad)



        return train_op, grad

    def build(self):
        self.add_placeholders()
        decoder_input= self.encode()
        self.start_pred, self.end_pred = self.decode(decoder_input)
        self.loss = self.add_loss_op(self.start_pred, self.end_pred)
        self.train_op, self.grad = self.add_train_op(self.loss)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" 
Training Seq2Seq model overhead code and generating answers (for now. should probably move to separate file)
"""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""




def sentence_to_token_ids(sentence, vocabulary, tokenizer=None):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    return [vocabulary.get(w, UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None):
    if True: # always do for now
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocab(vocabulary_path)
        with gfile.GFile(data_path, mode="rb") as data_file:
            with gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 5000 == 0:
                        print("tokenizing line %d" % counter)
                    token_ids = sentence_to_token_ids(line, vocab, tokenizer)
                    tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def process_glove(vocab_list, save_path, size=4e5):
    """
    :param vocab_list: [vocab]
    :return:
    """
    if True: # always do for now
        glove_path = os.path.join(FLAGS.glove_dir, "glove.840B.{}d.txt".format(FLAGS.embed_dim))
        glove = np.random.randn(len(vocab_list), FLAGS.embed_dim) # always do random initialization
        found = 0
        with open(glove_path, 'r') as fh:
            for line in tqdm(fh, total=size):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                if word in vocab_list:
                    idx = vocab_list.index(word)
                    glove[idx, :] = vector
                    found += 1
                if word.capitalize() in vocab_list:
                    idx = vocab_list.index(word.capitalize())
                    glove[idx, :] = vector
                    found += 1
                if word.lower() in vocab_list:
                    idx = vocab_list.index(word.lower())
                    glove[idx, :] = vector
                    found += 1
                if word.upper() in vocab_list:
                    idx = vocab_list.index(word.upper())
                    glove[idx, :] = vector
                    found += 1
                
        print("{}/{} of word vocab have corresponding vectors in {}".format(found, len(vocab_list), glove_path))
        np.savez_compressed(save_path, glove=glove)
        print("saved trimmed glove matrix at: {}".format(save_path))


def basic_tokenizer(sentence):
    words = []
    for space_separated_fragment in sentence.strip().split():
        words.extend(re.split(" ", space_separated_fragment))
    return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_paths, tokenizer=None):
    if True: # always do for now
        print("Creating vocabulary %s from data %s" % (vocabulary_path, str(data_paths)))
        vocab = {}
        for path in data_paths:
            with open(path, mode="rb") as f:
                counter = 0
                for line in f:
                    counter += 1
                    if counter % 100000 == 0:
                        print("processing line %d" % counter)
                    tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                    for w in tokens:
                        if w in vocab:
                            vocab[w] += 1
                        else:
                            vocab[w] = 1
        vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
        # limit to size FLAGS.vocab_size
       # vocab_list = vocab_list[:FLAGS.vocab_size]
        print("Vocabulary size: %d" % len(vocab_list))
        with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
            for w in vocab_list:
                vocab_file.write(w + b"\n")


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



def round_down(num, divisor):
    return num - (num%divisor)    



def gen_batches(epochs, batch_size, train_data):
    train_data_len = len(train_data)


    for i in range(epochs):
        np.random.shuffle(train_data)
        for ndx in tqdm(range(0, round_down(int(train_data_len),10), batch_size)):
            yield train_data[ndx:ndx+batch_size]





def train(model, train_data, val_data):
    tf.set_random_seed(2000)
    saver = tf.train.Saver()
    train_dir = FLAGS.train_dir
    train_data = [item  for item in train_data if item[2][1] < FLAGS.context_seq_length] # filter for training
    val_data = [item  for item in val_data if item[2][1] < FLAGS.context_seq_length] # filter for training

    val_data = val_data[:1000]
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
        logging.info("Loading model for generating answers")
        merged = tf.summary.merge_all()
        """
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                      sess.graph)
        """
        sess.run(tf.global_variables_initializer())


        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        training_losses = []
        val_losses = []
        min_val = float("inf")
      #  num_batches = len(train_data) / FLAGS.batch_size
        num_batches = int(round_down(int(len(train_data)),10) / FLAGS.batch_size)  # change to actual to compute epoch stats
        steps = 0
        avg_train_loss = 0
        avg_grad = 0
        epoch_training_losses = []
        run_val_losses = []
        train_step = 10
        val_step = 100
        for idx, train_batch in enumerate(gen_batches(FLAGS.epochs, FLAGS.batch_size, train_data)):
            steps += 1
           


            train_feed_dict = model.create_feed_dict(question_inputs= \
                np.array([item[0] for i,item in enumerate(train_batch) ]), \
                question_seq_lens=[np.where(item[0] == 0)[0][0] if item[0][-1] == 0 else FLAGS.question_seq_length for i,item in enumerate(train_batch)],
                context_inputs=np.array([item[1] for i,item in enumerate(train_batch) ]), \
                context_seq_lens=[np.where(item[1] == 0)[0][0] if item[1][-1] == 0 else FLAGS.context_seq_length for i,item in enumerate(train_batch)],
                start_decoder_labels=\
                 np.array([item[2][0] for i,item in enumerate(train_batch) ]),\
                 end_decoder_labels=\
                 np.array([item[2][1] for i,item in enumerate(train_batch) ]))
            summary, context_inputs, question_inputs, co_encode_decode_outputs, decoder_input, proj_question_outputs, context_outputs, question_outputs,\
            start_decode_outputs, end_decode_outputs, start_pred, end_pred, train_loss, _, grad = sess.run([merged, model.context_inputs, model.question_inputs,\
                model.co_encode_decode_outputs, model.decoder_input, model.proj_question_outputs, model.context_outputs, model.question_outputs, \
             model.start_decode_outputs,  model.end_decode_outputs, model.start_pred, model.end_pred, model.loss, model.train_op, model.grad], feed_dict=train_feed_dict)
            avg_train_loss += train_loss
            avg_grad += grad

            epoch_training_losses.append(train_loss)



        
            


            

            if (steps + 1) % train_step == 0:
                
                print("Training loss for previous batch", ":", avg_train_loss / train_step)
                print("And gradient: ",avg_grad / train_step)
                """
                print("Following is just for previous batch: ")
                print("context embeddings : ", context_inputs)
                print("question embeddings : ", question_inputs)
                print("And questions: ", train_feed_dict[model.question_input])
                print("And question sequence lengths: ", train_feed_dict[model.question_seq_lens])
                print("And context: ", train_feed_dict[model.context_input])
                print("And context sequence lengths: ", train_feed_dict[model.context_seq_lens])
                print("And question outputs : ", question_outputs)
                print("And projected question outputs : ", proj_question_outputs)
                print("And context outputs: ", context_outputs )
                print("And decoder input : ", decoder_input)
                print("And coencoding : ", co_encode_decode_outputs)
                print("And start_pred: ", start_pred)
                print("And labels: ", train_feed_dict[model.start_decoder_labels])
                print("And end_pred: ", end_pred)
                print("And labels: ", train_feed_dict[model.end_decoder_labels])
                print("And start decode outputs: ", start_decode_outputs)
                print("And end decode outputs: ", end_decode_outputs)
                training_losses.append(avg_train_loss)
                #train_writer.add_summary(summary)
                """
                avg_train_loss = 0
                avg_grad = 0

            if (idx + 1) % num_batches == 0:
                epoch = (idx + 1) / num_batches
                print("We finished epoch : ", str(epoch))
                avg_epoch_loss = np.sum(epoch_training_losses) / len(epoch_training_losses)
                print("Average epoch train loss was : ", str(avg_epoch_loss))
                training_losses.extend(epoch_training_losses)
                epoch_training_losses = []

            # val set test
            if (steps + 1) % val_step == 0:
                for idx, val_batch in enumerate(gen_batches(1, FLAGS.batch_size, val_data)):
                    val_feed_dict = model.create_feed_dict(question_inputs= \
                    np.array([item[0] for i,item in enumerate(val_batch)]), \
                    question_seq_lens=[np.where(item[0] == 0)[0][0] if item[0][-1] == 0 else FLAGS.question_seq_length for i,item in enumerate(val_batch)],
                    context_inputs=np.array([item[1] for i,item in enumerate(val_batch)]), \
                    context_seq_lens=[np.where(item[1] == 0)[0][0] if item[1][-1] == 0 else FLAGS.context_seq_length for i,item in enumerate(val_batch)],
                    start_decoder_labels=\
                     np.array([item[2][0] for i,item in enumerate(val_batch)]),\
                     end_decoder_labels=\
                     np.array([item[2][1] for i,item in enumerate(val_batch)]))
                    val_loss = model.loss.eval(feed_dict=val_feed_dict)
                    run_val_losses.append(val_loss)
                    print("Validation loss ", ":", val_loss)

                print("We finished validation!")
                avg_val_loss = np.sum(run_val_losses) / len(run_val_losses)
                print("Val loss of : ", str(avg_val_loss))
                if avg_val_loss < min_val:
                    min_val = avg_val_loss
                    print("New min val loss! ")
                    saver.save(sess, "train/model.ckpt")
                

                val_losses.extend(run_val_losses)
                run_val_losses = []
                





        return training_losses, val_losses

 
   
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
    dev_question_data = dataset[0]
    dev_context_data = dataset[1]
    dev_question_uuid_data = dataset[2]
    dev_len = len(dev_context_data)
    start_predictions = np.array([])
    end_predictions = np.array([])

    do = False
    # split b/c gpu memory limitations...
    split = 20
    for i in xrange(split):
        start_index = int(i*(dev_len / split))
        end_index = int((i+1) * (dev_len / split))
        question_input_data = dev_question_data[start_index:end_index]
        context_input_data = dev_context_data[start_index:end_index]
        feed_dict = model.create_feed_dict(question_inputs=question_input_data, \
                question_seq_lens=[np.where(item == 0)[0][0] if item[-1] == 0 else FLAGS.question_seq_length for i,item in enumerate(question_input_data)],
                context_inputs=context_input_data, \
                context_seq_lens=[np.where(item == 0)[0][0] if item[-1] == 0 else FLAGS.context_seq_length for i,item in enumerate(context_input_data)])
        new_start_predictions, new_end_predictions = sess.run([model.start_pred, model.end_pred], feed_dict=feed_dict)
        start_predictions = np.append(start_predictions,[new_start_predictions])
        end_predictions = np.append(end_predictions,[new_end_predictions])

    

    start_predictions = start_predictions.reshape(dev_len, FLAGS.context_seq_length)
    end_predictions = end_predictions.reshape(dev_len, FLAGS.context_seq_length)

    predictions_len = len(start_predictions) 



    for i in xrange(predictions_len):
      
        start_prediction = start_predictions[i]
        end_prediction = end_predictions[i]
        start_ans_span = np.argmax(start_prediction)
        end_ans_span = np.argmax(end_prediction)
        answer = ""
        for j in xrange(start_ans_span, end_ans_span + 1):
            word_index = dev_context_data[i][j]
            word = rev_vocab[int(word_index)].decode("utf8")
            answer += (word + " ")

        answer = answer[:-1] if answer else answer
        answers[dev_question_uuid_data[i]] = answer

    return answers

""" Handle overall training process for seq2Seq baseline model
"""
def main(_):

     # create vocab
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    """
    create_vocabulary(vocab_path,
                      [pjoin(FLAGS.data_dir, "train.context"),
                       pjoin(FLAGS.data_dir, "train.question"),
                       pjoin(FLAGS.data_dir, "val.context"),
                       pjoin(FLAGS.data_dir, "val.question")])
    """
    vocab, rev_vocab = initialize_vocab(vocab_path)


    #define pre-trained embeddings
    """
    process_glove(rev_vocab, pjoin(FLAGS.data_dir, "glove.trimmed.{}".format(FLAGS.embed_dim)))
    """
    # load embeddings 
    embed_path = FLAGS.embed_path or pjoin("data", "squad", \
        "glove.trimmed.{}.npz".format(FLAGS.embed_dim))
    embed_vals = np.load(embed_path)
    pretrained_embeddings = embed_vals['glove']
   
    """
    # create dataset
    data_to_token_ids(pjoin(FLAGS.data_dir, "train.context"),pjoin(FLAGS.data_dir, "train.ids.context"), vocab_path)
    data_to_token_ids(pjoin(FLAGS.data_dir, "train.question"),pjoin(FLAGS.data_dir, "train.ids.question"), vocab_path)
    data_to_token_ids(pjoin(FLAGS.data_dir, "val.context"),pjoin(FLAGS.data_dir, "val.ids.context"), vocab_path)
    data_to_token_ids(pjoin(FLAGS.data_dir, "val.question"),pjoin(FLAGS.data_dir, "val.ids.question"), vocab_path)

    """

    # download and save training data for later
    """
    train_question_data, train_context_data, train_answer_spans = \
        read_train_dataset(pjoin(FLAGS.data_dir, FLAGS.train_question_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.train_context_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.train_answer_path))
    train_data_path = FLAGS.train_data_path
    np.savez_compressed(train_data_path, question_data=train_question_data, context_data=train_context_data, \
        answer_spans=train_answer_spans)
    # download and save val data for later
    val_question_data, val_context_data, val_answer_spans = \
        read_train_dataset(pjoin(FLAGS.data_dir, FLAGS.val_question_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.val_context_ids_path), \
            pjoin(FLAGS.data_dir,FLAGS.val_answer_path))
    val_data_path = FLAGS.val_data_path
    np.savez_compressed(val_data_path, question_data=val_question_data, context_data=val_context_data,\
        answer_spans=val_answer_spans)
    """
    # load train data

   
    train_data_path = FLAGS.train_data_path
    loaded_train_data = np.load(train_data_path)
    train_question_data = loaded_train_data['question_data']
    train_context_data = loaded_train_data['context_data']
    train_answer_spans = loaded_train_data['answer_spans']




    # load val data
    val_data_path = FLAGS.val_data_path
    loaded_val_data = np.load(val_data_path)
    val_question_data = loaded_val_data['question_data']
    val_context_data = loaded_val_data['context_data']
    val_answer_spans = loaded_val_data['answer_spans']




   
 

    train_data = zip(train_question_data, train_context_data, train_answer_spans)
    val_data = zip(val_question_data, val_context_data, val_answer_spans)
    

     # initialize model
    seq2Seq= Seq2Seq(pretrained_embeddings=pretrained_embeddings)

    train(seq2Seq, train_data, val_data)

 
    # generate answers on dev set from best saved model

    
    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    dev_question_data, dev_context_data, dev_question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    np.savez_compressed(FLAGS.dev_data_path, context_data=dev_context_data,question_data=dev_question_data, question_uuid_data=dev_question_uuid_data)

    # load dev data
    dev_data_path = FLAGS.dev_data_path
    loaded_dev_data = np.load(dev_data_path)
    dev_question_data = loaded_dev_data['question_data']
    dev_context_data = loaded_dev_data['context_data']
    dev_question_uuid_data = loaded_dev_data['question_uuid_data']

    dev_dataset = (dev_question_data, dev_context_data, dev_question_uuid_data)
    answers = {}

    # note: need to load model/variables, then saver, and then restore if you would like
    seq2Seq= Seq2Seq(pretrained_embeddings=pretrained_embeddings)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        logging.info("Loading model for generating answers")
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'train/model.ckpt')
        answers = generate_answers(sess, seq2Seq, dev_dataset ,rev_vocab)
        #answers = generate_answers(sess, seq2Seq, train_dataset ,rev_vocab)


  

    #write to json file to root dir
    with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:

        f.write(unicode(json.dumps(answers, ensure_ascii=False)))

    
    
if __name__ == "__main__":
    tf.app.run()