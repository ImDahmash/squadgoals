from keras.layers import *
from keras.models import *
from keras.preprocessing import *

GLOVE_PATH = "data/squad/glove.squad.100d.npy"
TRAIN_PATH = "data/squad/train.npz"
VAL_PATH = "data/squad/val.npz"

# Load the GloVe vectors
glove = np.load(GLOVE_PATH)
train_data = np.load(TRAIN_PATH)
arr_q = train_data["question"]
arr_c = train_data["context"]
sparse_y = np.expand_dims(train_data["answer"], -1)

max_c_len = arr_c.shape[1]
max_q_len = arr_q.shape[1]

in_q = Input(shape=arr_q.shape, dtype="int32", name="question")
in_c = Input(shape=arr_c.shape, dtype="int32", name="context")

# Embed the question and answer
x_c = Embedding(input_dim=glove.shape[0],
                output_dim=glove.shape[1],
                input_length=max_c_len,
                mask_zero=True,
                weights=[glove])(in_c)

x_q = Embedding(input_dim=glove.shape[0],
                output_dim=glove.shape[1],
                input_length=max_q_len,
                mask_zero=True,
                weights=[glove])(in_q)

# Encoder: Takes in the following

concat_embeddings = merge([x_q, x_c], mode='concat', concat_axis=1)
lstm = LSTM(200)(concat_embeddings)
rev_lstm = LSTM(200, go_backwards=True)(concat_embeddings)
merged = merge([lstm, rev_lstm], mode='concat')
result = Dense(max_c_len, activation='softmax')(merged)

model = Model(input=[in_c, in_q], output=result)


# Build the model
model = Model(input=inputs, output=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

model.fit((arr_c, arr_q), sparse_y)

