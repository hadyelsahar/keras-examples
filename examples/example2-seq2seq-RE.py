
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('./data/google-re.csv')
X = df.sentence.values


TOP_WORDS = 1000
MAX_SEQ_LENGTH = 1000
EMB_VEC_LENGTH = 50
N_EPOCHS = 5

tokenizer = Tokenizer(nb_words=TOP_WORDS)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)

y = df.pred.values
encoder = LabelEncoder()
encoder.fit(y)
y = encoder.transform(y)
y = np_utils.to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = sequence.pad_sequences(X_train, maxlen=MAX_SEQ_LENGTH)
X_test = sequence.pad_sequences(X_test, maxlen=MAX_SEQ_LENGTH)


# ---------------------------------------------------#
# First toy model try LSTM + Relation classification #
# ---------------------------------------------------#

model = Sequential()
model.add(Embedding(TOP_WORDS, EMB_VEC_LENGTH, input_length=MAX_SEQ_LENGTH))
model.add(LSTM(100, dropout_U=-0.2, dropout_W=0.2))
model.add(Dense(len(y[0]), activation='relu'))
sgd = SGD(lr=0.01, clipnorm=1.)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy', 'fmeasure'])
print(model.summary())

model.fit(X_train, y_train, nb_epoch=N_EPOCHS, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test)
print("\n Model: LSTM: Accuracy: %.2f%%" % (scores[1]*100))
