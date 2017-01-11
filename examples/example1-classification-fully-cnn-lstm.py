import numpy as np
from keras.datasets import imdb
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Conv1D, MaxPooling1D, Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import SGD
from gensim.models import Word2Vec
from os import path

# fix random seed for reproducibility
np.random.seed(7)

####################
# Loading Datasets #
####################
# IMDB Sentiment Classification
# load the dataset but only keep the top n words, zero the rest
TOP_WORDS = 5000
MAX_SEQ_LENGTH = 500
EMB_VEC_LENGTH = 150
N_EPOCHS = 30

_W2V_BINARY_PATH = path.dirname(path.abspath(__file__)) + "/../data/wordvectors/GoogleNews-vectors-negative300.bin.gz"
word_vectors = Word2Vec.load_word2vec_format(_W2V_BINARY_PATH, binary=True)

(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=TOP_WORDS)
X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LENGTH)

# #########################
# # model1 : one layer NN #
# #########################
# # model http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
#
# model = Sequential()
# model.add(Dense(output_dim=64, input_dim=MAX_SEQ_LENGTH))
# model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
# model.add(Dense(8, init='uniform', activation='relu'))
# model.add(Dense(1, init='uniform', activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True), metrics=['accuracy'])
# # print(model.summary())
#
# model.fit(X_train, y_train, batch_size=32, nb_epoch=N_EPOCHS)
# scores = model.evaluate(X_test, y_test, batch_size=32)
# print("\n Model1: One layer NN: Accuracy: %.2f%%" % (scores[1]*100))

#################
# model2 : LSTM #
#################
# model from : http://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# more about LSTM :
# http://deeplearning.net/tutorial/lstm.html
# http://colah.github.io/posts/2015-08-Understanding-LSTMs/

# create the model
model = Sequential()
model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, input_length=MAX_SEQ_LENGTH))
model.add(LSTM(100, consume_less='gpu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.fit(X_train, y_train, nb_epoch=N_EPOCHS, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test)
print("\n Model2: LSTM: Accuracy: %.2f%%" % (scores[1]*100))
#
# ######################
# # model3: CNN + LSTM #
# ######################
# EMB_VEC_LENGTH = 32
# model = Sequential()
# model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, input_length=MAX_SEQ_LENGTH))
# model.add(Conv1D(nb_filter=32, filter_length=3, activation='relu', border_mode='same'))
# model.add(MaxPooling1D(pool_length=2))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # print model.summary()
# model.fit(X_train, y_test, batch_size=64, nb_epoch=N_EPOCHS)
#
# scores = model.evaluate(X_test, y_test)
# print("\n Model3: CNN+ LSTM + DROPOUT: Accuracy: %.2f%%" % (scores[1]*100))

# #################################
# # model3: CNN + LSTM + Word2vec #
# #################################
#
# index_dict = imdb.get_word_index()
# embedding_weights = np.zeros((TOP_WORDS + 1, word_vectors.vector_size))
#
# for word, index in index_dict.items():
#     if word in word_vectors and index <= TOP_WORDS:
#         embedding_weights[index, :] = word_vectors[word]
#
# model = Sequential()
# model.add(Embedding(TOP_WORDS + 1, word_vectors.vector_size, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights]))
# model.add(Conv1D(nb_filter=32, filter_length=3, activation='relu', border_mode='same', input_shape=[MAX_SEQ_LENGTH, word_vectors.vector_size]))
# model.add(MaxPooling1D(pool_length=2))
# model.add(Dropout(0.2))
# model.add(LSTM(100))
# model.add(Dropout(0.2))
# model.add(Dense(1, activation='relu'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# # print model.summary()
# model.fit(X_train, y_test, batch_size=64, nb_epoch=N_EPOCHS)
#
# scores = model.evaluate(X_test, y_test)
# print("\n Model4: CNN + LSTM + Word2Vec: Accuracy: %.2f%%" % (scores[1]*100))

#################################
# model4: LSTM + Word2vec #
#################################

index_dict = imdb.get_word_index()

embedding_weights = np.zeros((TOP_WORDS + 1, word_vectors.vector_size))
for word, index in index_dict.items():
    if word in word_vectors and index <= TOP_WORDS:
        embedding_weights[index, :] = word_vectors[word]

model = Sequential()
model.add(Embedding(TOP_WORDS + 1, word_vectors.vector_size, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights]))
model.add(LSTM(100, consume_less='gpu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# print model.summary()
model.fit(X_train, y_test, batch_size=64, nb_epoch=N_EPOCHS)

scores = model.evaluate(X_test, y_test)
print("\n Model4: LSTM + WORDVECTOR: Accuracy: %.2f%%" % (scores[1]*100))

#################################
# model5: BidirectionalLSTM + Word2vec #
#################################

index_dict = imdb.get_word_index()

embedding_weights = np.zeros((TOP_WORDS + 1, word_vectors.vector_size))
for word, index in index_dict.items():
    if word in word_vectors and index <= TOP_WORDS:
        embedding_weights[index, :] = word_vectors[word]

model = Sequential()
model.add(Embedding(TOP_WORDS + 1, word_vectors.vector_size, input_length=MAX_SEQ_LENGTH, weights=[embedding_weights]))
model.add(Bidirectional(LSTM(100, consume_less='gpu')))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print model.summary()
model.fit(X_train, y_test, batch_size=64, nb_epoch=N_EPOCHS)

scores = model.evaluate(X_test, y_test)
print("\n MODEL5: BIDIRECTION LSTM + WORDVECTOR: Accuracy: %.2f%%" % (scores[1]*100))