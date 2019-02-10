import pandas as pd
from keras import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.text import text_to_word_sequence
from keras_preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import numpy as np
encoder = LabelBinarizer(sparse_output=False)

vocab = 4000
max_len = 100
documents = pd.read_csv("JobClassification/example-data/data.csv")
# titles = documents["title"].values.astype(str)

employmentType = documents["employmentType"].values.astype(str)
num_classes = len(set(employmentType))

descriptions = documents["description"].values.astype(str)

# get vocab size
# print(len(tokenizer.word_index)+1)
train_size = int(len(descriptions) * .8)

x_train_texts = descriptions[:train_size]
y_train = list(employmentType[:train_size])
x_test_texts = descriptions[train_size:]
y_test = list(employmentType[train_size:])


tokenizer = Tokenizer(num_words=vocab)
tokenizer.fit_on_texts(x_train_texts)

# find max length
# max_len = max([len(requirement.split()) for requirement in requirements])
# print(max_len)
x_train = tokenizer.texts_to_matrix(x_train_texts, mode="tfidf")
x_test = tokenizer.texts_to_matrix(x_test_texts, mode="tfidf")

# data_sequences = pad_sequences(data, maxlen=max_len, padding="post")

# x_train, x_test, y_train, y_test = train_test_split(data, labels, train_size=0.8)
encoder.fit(y_train+y_test)
y_train = encoder.fit_transform(y_train)
y_test = encoder.fit_transform(y_test)


# x_train_tokens = tokenizer.texts_to_matrix(x_train_texts, mode="tfidf")
# x_test_tokens = tokenizer.texts_to_matrix(x_test_texts, mode="tfidf")

# x_train = pad_sequences(x_train_tokens, maxlen=max_len, padding="post")
# x_test = pad_sequences(x_test_tokens, maxlen=max_len, padding="post")

print(x_train.shape[1], y_train.shape[1])
# print(x_test.shape[1], y_test.shape[1])

model = Sequential()
model.add(Dense(256, input_shape=(vocab,)))
model.add(Activation('relu'))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=32,
                    epochs=10,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

