"""
Perform training and evaluation of an LSTM model over the dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import *
from keras.models import *
from keras.objectives import *

MAX_SEQUENCE_LENGTH = 1000  # Max number of tokens in a sequence

vocab = {}
with open("data/squad/vocab.dat", "r") as fh:
    i = 0
    for line in fh:
        vocab[line.strip()] = i
        i += 1


def sentence_to_ids(sentence):
    token_ids = []
    for tok in sentence.split():
        token_ids.append(vocab.get(tok, 2))
    return token_ids

sequences = list(map(sentence_to_ids, [
    "I love ice cream",
    "I hate steamed broccoli",
    "I love my family",
    "I hate my enemies",
    "I hate Trump",
]))

# Try and train a network for sentence completion.
begins = list(map(lambda s: s[:-1], sequences))
ends = list(map(lambda s:s[-1], sequences))

embeddings_matrix = np.load("data/squad/glove.trimmed.100.npz")["glove"]  # Load embedding matrix
embedding_layer = Embedding(embeddings_matrix.shape[0], embeddings_matrix.shape[1],   # Embeddings
                            weights=[embeddings_matrix],
                            trainable=False,
                            name="Embedding")

input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name="input")
embed_seq = embedding_layer(input, name="embed_seq")
preds = Dense(embeddings_matrix.shape[0], activation="softmax")(embed_seq)  # Predict based on number of tokens

model = Model(input, preds)
model.compile(loss=mse, optimizer="adam", metrics=["acc"])
model.fit(begins, ends)