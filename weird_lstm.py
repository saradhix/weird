'''Trains a LSTM on the IMDB sentiment classification task.
The dataset is actually too small for LSTM to be of any advantage
compared to simpler, much faster methods such as TF-IDF + LogReg.
Notes:

- RNNs are tricky. Choice of batch size is important,
choice of loss and optimizer is critical, etc.
Some configurations won't converge.

- LSTM loss decrease patterns during training can be quite different
from what you see with CNNs/MLPs/etc.
'''
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import glob
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.datasets import imdb
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from nltk.tokenize import TweetTokenizer
import sys
import os
from sklearn.cross_validation import train_test_split

import json
import libspacy


#labels = to_categorical(np.asarray(labels))
def main():
  weird_news = 'weird.json'
  normal_news='normal3.json'

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


  #Create the test and training sets
  #shuffle(raw_weird)
  #shuffle(raw_normal)

  train = 7000
  test=1000

  X_raw_train=raw_weird[:train]+raw_normal[:train]
  X_raw_test=raw_weird[train:train+test]+raw_normal[train:train+test]
  #Create the labels 1st half are weird class 1, rest are normal 0
  y_train=[1]*train+[0]*train
  y_test=[1]*test+[0]*test

  X_train=[]
  X_test=[]

  print("Extracting features from train")
  for raw_title in X_raw_train:
    features = generate_features(raw_title)
    #print(features)
    #sys.exit()
    X_train.append(features)

  print( "Extracting features from test")
  for raw_title in X_raw_test:
    features = generate_features(raw_title)
    X_test.append(features)

  print('Building model...')
  model = Sequential()
  model.add(LSTM(128,input_dim=len(features), dropout_W=0.2, dropout_U=0.2))  # try using a GRU instead, for fun
  model.add(Dense(2))
  model.add(Activation('sigmoid'))

# try using different optimizers and different optimizer configs
  model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

  print('Train...')
  model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
  score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
  print('Test score:', score)
  print('Test accuracy:', acc)

  predicted = model.predict_classes(np.array(data_test),batch_size=batch_size)
  print(predicted.shape)
  #predicted = np.reshape(predicted, (predicted.size,))
  #print predicted.shape
def generate_features(title):
  features=[]
  features=libspacy.get_vector(title)
  return features.tolist()

if __name__ == "__main__":
  main()
