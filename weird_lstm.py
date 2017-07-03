import json
import sys
from random import shuffle
import os
import re
import sys
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text

def main():
  print("Hello")
  seed = 0
  np.random.seed(seed)

  weird_news='upi_processed.json'
  normal_news='normal_jumbo.json'

  raw_weird=[]
  raw_normal=[]
  #Load weird news
  fd = open(weird_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_weird.append(str(title))

  print( "Weird news items :", len(raw_weird))

  #Load normal news
  fd = open(normal_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_normal.append(str(title))

  print( "Normal news items :", len(raw_normal))

  train = 5000
  test=1000

  X_raw_train=raw_weird[:train]+raw_normal[:train]
  X_raw_test=raw_weird[train:train+test]+raw_normal[train:train+test]
  #Create the labels 1st half are weird class 1, rest are normal 0
  y_train=[1]*train+[0]*train
  y_test=[1]*test+[0]*test

  print( "Training :", len(y_train), "Testing :", len(y_test))
  max_len = 80

  tk = text.Tokenizer(num_words=200000)
  tk.fit_on_texts(X_raw_train+X_raw_test)
  X_train = tk.texts_to_sequences(X_raw_train)
  X_train = sequence.pad_sequences(X_train, maxlen=max_len)
  print("Dimensions of X_train",X_train.shape)
  X_test = tk.texts_to_sequences(X_raw_test)
  X_test = sequence.pad_sequences(X_test, maxlen=max_len)
  word_index = tk.word_index
  ytrain_enc = np_utils.to_categorical(y_train)

  model = Sequential()
  model.add(Embedding(len(word_index) + 1, 300, input_length=max_len, dropout=0.2))
  model.add(LSTM(300, dropout=0.2, recurrent_dropout=0.2))

  model.add(Dense(200))
  model.add(PReLU())
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Dense(200))
  model.add(PReLU())
  model.add(Dropout(0.2))
  model.add(BatchNormalization())

  model.add(Dense(2))
  model.add(Activation('softmax'))
  model.summary()

  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  checkpoint = ModelCheckpoint('weights.h5', monitor='val_acc', save_best_only=True, verbose=2)

  model.fit(X_train, y=ytrain_enc,
                 batch_size=128, epochs=20, validation_split=0.1,
                 shuffle=True, callbacks=[checkpoint])




#===========================
if __name__ == "__main__":
  main()
