import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Conv1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import SGD



# fix random seed for reproducibility
np.random.seed(7)

####################
# Loading Datasets #
####################
# IMDB Sentiment Classification
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
max_seq_length = 500
embedding_vector_length = 150


(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
X_train = sequence.pad_sequences(X_train, maxlen=max_seq_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_seq_length)


#########################
# model1 : one layer NN #
#########################
# model http://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/
model = Sequential()
X_fully_tain = np.expand_dims(X_train, axis=1)
X_fully_test = np.expand_dims(X_test, axis=1)
# model.add(Embedding(top_words, embedding_vector_length, input_length=max_seq_length))

model.add(Dense(output_dim=64, input_dim=max_seq_length))
model.add(Dense(12, input_dim=8, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01, momentum=0.9, nesterov=True))

model.fit(X_train, y_train, batch_size=32, nb_epoch=10)
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=32)
print loss_and_metrics





