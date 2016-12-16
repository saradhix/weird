import json
import sys
import nltk
from nltk.corpus import stopwords
import libspacy
import libgrams
import libwordnet
from random import shuffle
import os
import re
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
def main():
  print "Hello"
  seed = 0 
  numpy.random.seed(seed)

  weird_news='weird.json'
  normal_news='normal.json'

  raw_weird=[]
  raw_normal=[]
  #Load weird news
  fd = open(weird_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_weird.append(str(title))

  print "Weird news items :", len(raw_weird)

  #Load normal news
  fd = open(normal_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_normal.append(str(title))

  print "Normal news items :", len(raw_normal)
#  '''
  print_num_articles_with_colon(raw_weird)
  print_num_articles_with_colon(raw_normal)
  print_num_articles_with_exclam(raw_weird)
  print_num_articles_with_exclam(raw_normal)

  print_num_articles_with_country(raw_weird)
  print_num_articles_with_country(raw_normal)
  print_sentence_structure(raw_weird)
  print_sentence_structure(raw_normal)
  print "-"*40
  print_num_stop_words(raw_weird)
  print_num_stop_words(raw_normal)
  print "-"*40
  print_avg_word_len(raw_weird)
  print_avg_word_len(raw_normal)
  print "-"*40
  print_pos_distributions(raw_weird)
  print_pos_distributions(raw_normal)
  print "-"*40
  print_quoted_counts(raw_weird)
  print_quoted_counts(raw_normal)
  print "-"*40
  most_repeated_bigrams(raw_weird)
  most_repeated_bigrams(raw_normal)
  print "-"*40
  most_rep_sub_w = most_repeated_subjects(raw_weird)
  print "Most repeated subjects in weird", most_rep_sub_w
  print_num_articles_with_popular_subject(raw_weird, most_rep_sub_w)

  most_rep_sub_n = most_repeated_subjects(raw_normal)
  print_num_articles_with_popular_subject(raw_normal, most_rep_sub_n)
  print "Most repeated subjects in normal", most_rep_sub_n
  print "-"*40
  avg_capitalized_words(raw_weird)
  avg_capitalized_words(raw_normal)
  print "-"*40
  avg_animals(raw_weird)
  avg_animals(raw_normal)
  print "-"*40
  avg_body_parts(raw_weird)
  avg_body_parts(raw_normal)
  print "-"*40
  avg_nes(raw_weird)
  avg_nes(raw_normal)
  print "-"*40
  avg_nes_halves(raw_weird)
  avg_nes_halves(raw_normal)
  print "-"*40
  nvn_phrases(raw_weird)
  nvn_phrases(raw_normal)
  print "-"*40
#  '''

  #Generate the top N verbs
  top_weird_verbs = get_top_verbs(raw_weird)
  print "Top weird verbs =", top_weird_verbs
  top_normal_verbs = get_top_verbs(raw_normal)
  print "Top normal verbs =", top_normal_verbs

  print_len_distributions(raw_weird)
  print_len_distributions(raw_normal)
  sys.exit()
  #Create the test and training sets
  shuffle(raw_weird)
  shuffle(raw_normal)

  train = 7000
  test=1000

  X_raw_train=raw_weird[:train]+raw_normal[:train]
  X_raw_test=raw_weird[train:train+test]+raw_normal[train:train+test]
  #Create the labels 1st half are weird class 1, rest are normal 0
  y_train=[1]*train+[0]*train
  y_test=[1]*test+[0]*test

  X_train=[]
  X_test=[]

  print "Extracting features from train"
  for raw_title in X_raw_train:
    features = generate_features(raw_title)
    #print features
    X_train.append(features)

  print "Extracting features from test"
  for raw_title in X_raw_test:
    features = generate_features(raw_title)
    X_test.append(features)

  num_features = len(features)
  print "Size of train, test", len(X_train), len(X_test)
  print "Size of  labels train, test", len(y_train), len(y_test)
  print "#features=", num_features

  #Start training a neural network
  model = Sequential()
  model.add(Dense(100, input_dim=len(features), init='uniform' ))
  model.add(Activation('sigmoid'))
  model.add(Dropout(0.3))
#  model.add(Dense(140, init='uniform', activation='relu'))
  model.add(Dense(80, init='uniform'))
  model.add(Activation('tanh'))
  model.add(Dropout(0.2))
  model.add(Dense(40, init='uniform'))
  model.add(Activation('sigmoid'))
  model.add(Dropout(0.2))
  model.add(Dense(1, init='uniform', activation='sigmoid'))
  # Compile model
  model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
  # Fit the model
  model.fit(X_train, y_train, nb_epoch=150, batch_size=100)
  # evaluate the model
  scores = model.evaluate(X_train, y_train)
  print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
  print "Running predictions "
  y_pred = model.predict(X_test)
  y_pred =[ int(round(i)) for i in y_pred]
  print confusion_matrix(y_test, y_pred)
  print classification_report(y_test, y_pred)

  for i, (actual, predicted) in enumerate(zip(y_test, y_pred)):
    if actual != predicted:
      print "Actual=", actual, "Predicted=", predicted
      print X_raw_test[i]
      print X_test[i]

  print confusion_matrix(y_test, y_pred)
  print classification_report(y_test, y_pred)
  sys.exit()
  #Now try with SVM with RBF kernel
  C = 1.0  # SVM regularization parameter
  rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train, y_train)
  y_pred = rbf_svc.predict(X_test)
  print confusion_matrix(y_test, y_pred)
  print classification_report(y_test, y_pred)

synonyms=['weird', 'supernatural', 'unearthly', 'strange', 'abnormal', 'unusual',
        'uncanny', 'eerie', 'unnatural', 'unreal', 'ghostly', 'mysterious',
        'magical', 'bizarre', 'peculiar', 'funny', 'curious', 'offbeat',
        'eccentric', 'unconventional', 'unorthodox', 'abnormal', 'atypical',
        'puzzling', 'deviant', 'aberrant', 'grotesque', 'surreal']

freq_verbs1 = ['say', 'get', 'found', 'arrest', 'offer', 'change', 'find', 'accuse',
        'sue', 'win', 'take', 'jail', 'make', 'seek', 'pay', 'give', 'caught', 
        'stolen', 'lose', 'turn', 'survive', 'seek', 'give', 'stolen', 'lose', 
        'turn', 'ask', 'face', 'die', 'kill', 'apologize', 'end', 'save', 'sell',
        'speak', 'eat', 'driv', 'ban', 'leave', 'keep', 'win', 'held', 'put', 
        'come', 'steal', 'stop', 'try', 'nab', 'ban', 'miss', 'sentence', 'sue',
        'fire', 'protest', 'plead', 'call', 'slam', 'visit', 'hit', 'reject',
        'accuse', 'join', 'seek', 'offer', 'led', 'discuss', 'need', 'leave',
        'bring', 'arrest', 'fight', 'continue', 'announce', 'protest', 'vow',
        'condemn', 'refuse', 'write', 'delay', 'slash', 'ready', 'prepar', 'deal'
        'sign', 
        ]

freq_verbs = ['say', 'found', 'arrest',  'accuse',
        'sue', 'jail', 'caught',
        'stolen', 'lose', 'survive',
        'die', 'kill', 'apologize', 'end', 'save',
        'eat', 'driv', 'ban', 'leave', 'keep', 'win',
        'steal', 'stop', 'nab', 'ban', 'miss', 'sentence', 'sue',
        'fire', 'protest', 'plead', 'call', 'slam', 'visit', 'hit', 'reject',
        'accuse', 'join', 'seek', 'offer', 'led', 'discuss', 'need', 'leave',
        'bring', 'fight', 'continue', 'announce', 'protest', 'vow',
        'condemn', 'refuse', 'write', 'wound', 'lead', 'member','quit',
        'withdraw',
        'porn', 'sex', 'condom', 'nude', 'dead', 'male', 'female', 'mistake'
        'sink', 'food', 'sky', 'auction', 'pay', 'forgot',  ]
countries = []
def generate_features(title):
  features=[]
  #First feature is the sentence structure ie words in the title
  words=title.split(' ')
  num_words = len(words)
  features.append(num_words)

  #Number of stop words
  stop_bool = [ 1 if w in stopwords.words("english") else 0 for w in words]
  num_stop = sum(stop_bool)
  features.append(num_stop)

  #Average word length
  avg_word_len = float(len(title))/num_words
  features.append(avg_word_len)

  #Pos counts
  pos_counts = libspacy.get_pos_counts(str(title))
  features +=pos_counts

  #Quoted characters
  num_quotes = title.count("'")
  if num_quotes >=2:
    num_quotes = 1
  features.append(num_quotes)

  #Num capitalized words
  num_cap_words =sum( [word.upper()==word and word.lower() !=word for word in words])
  features.append(num_cap_words)

  #Number of animals and human body parts
  nouns = libspacy.get_nouns(title)
  animals = [w for w in nouns if libwordnet.is_animal(w) ]
  num_animals = len(animals)
  parts = [w for w in nouns if libwordnet.is_body_part(w) ]
  num_parts = len(animals)
  features.append(num_animals)
  features.append(num_parts)

  #NEs in first and second halves
  total_f = total_s = 0
  nes = libspacy.get_nes(' '.join(title.split(' ')[:len(title.split(' '))/2]))
  total_f +=len(nes)
  nes = libspacy.get_nes(' '.join(title.split(' ')[len(title.split(' '))/2:]))
  total_s +=len(nes)

  features.append(total_f)
  features.append(total_s)

  #Presence of colon character

  colon = title.count(':')
  features.append(colon)

  f_syn = 0
  for syn in synonyms:
    if syn in words:
      f_syn = 1
      break

  features.append(f_syn)

  for verb in freq_verbs:
    if verb in title.lower():
      features.append(1)
    else:
      features.append(0)
  if '!' in title:
    features.append(1)
  else:
    features.append(0)
  if '?' in title:
    features.append(1)
  else:
    features.append(0)
  if '-' in title:
    features.append(1)
  else:
    features.append(0)
#Country as feature
  count =0
  for country in countries:
    if country in title.lower():
      count = count+1
  features.append(count)
  return features

def print_sentence_structure(titles):
  total_words = 0
  num_titles = len(titles)
  for title in titles:
    words = title.split(' ')
    total_words = total_words + len(words)

  words_per_title = float(total_words) / num_titles
  print "Words per title=", words_per_title


def print_num_articles_with_colon(titles):
  count =0
  num_titles = len(titles)
  for title in titles:
    if ':' in title:
      count +=1
  print 1.0 * count / num_titles


def print_num_articles_with_exclam(titles):
  count =0
  num_titles = len(titles)
  for title in titles:
    if '!' == title.strip()[-1]:
      count +=1
  print "Exclaim", 1.0 * count / num_titles

def print_num_articles_with_country(titles):
  count =0
  num_titles = len(titles)
  for title in titles:
    words = title.lower().strip().split(' ')
    for country in countries:
       if country in words:
         count +=1
  print "Country count=", count, "Percent=", 1.0 * count / num_titles

def print_num_stop_words(titles):
  total_words = 0
  num_titles = len(titles)
  for title in titles:
    words = title.split(' ')
    stop_bool = [ 1 if w in stopwords.words("english") else 0 for w in words]
    total_words = total_words + sum(stop_bool)

  words_per_title = float(total_words) / num_titles
  print "Stop Words per title=", words_per_title


def print_avg_word_len(titles):
  total_words = 0
  total_word_lengths = 0
  num_titles = len(titles)
  for title in titles:
    words = title.split(' ')
    total_words += len(words)
    total_word_lengths += len(title)

  avg_word_len = float(total_word_lengths) / total_words
  print "Average word length=", avg_word_len


def print_pos_distributions(titles):
  total_counts = [0,0,0,0,0,0,0]
  num_titles = len(titles)
  for title in titles:
    #print title
    title_counts = libspacy.get_pos_counts(str(title))
    total_counts = [ m+n for (m,n) in zip(total_counts, title_counts)]

  total_counts=map(lambda x:float(x)/num_titles, total_counts)

  print "POS_stats", total_counts

def print_quoted_counts(titles):
  total_num_quotes = 0
  num_titles = len(titles)
  for title in titles:
    num_quotes = title.count("'")
    if num_quotes >=2:
      total_num_quotes +=1
  print "Quoted count=", total_num_quotes
  print "Average quotes=", 1.0*total_num_quotes/num_titles

def most_repeated_bigrams(titles):
  bigrams={}

  for title in titles:
   grams = libgrams.make_trigrams(title.lower())
   for gram in grams:
     bigrams[gram]=bigrams.get(gram,0)+1

  #for gram in bigrams:
    #print gram, bigrams[gram]
  #print "-"*40
  #for tup in sorted(bigrams.items(), key=lambda x:x[1], reverse=True)[:30]:
  #  print tup
  #sys.exit()

def most_repeated_subjects(titles):
  subjects={}

  for title in titles:
    nsubs = libspacy.get_nsubj(title.lower())
    for nsub in nsubs:
      nsub = str(nsub)
      count=subjects.get(nsub,0)+1
      subjects[nsub]=subjects.get(nsub,0)+1
  frq =[]
  for tup in sorted(subjects.items(), key=lambda x:x[1], reverse=True)[0:40]:
    #print tup
    frq.append(tup[0])
  print "**"*40
  return frq


def avg_capitalized_words(titles):
  total_caps = 0
  num_titles = len(titles)
  for title in titles:
    for word in title.split(' '):
      if word.upper()==word and word.lower() !=word and len(word) >4:
        #print word, title
        total_caps +=1

  avg_caps = float(total_caps)/num_titles
  print "Capitalized words", avg_caps

def avg_animals(titles):
  total_animals = 0
  num_titles = len(titles)
  for title in titles:
    nouns = libspacy.get_nouns(title)
    animals = [w for w in nouns if libwordnet.is_animal(w) ]
    #if animals:
    #  print animals
    total_animals +=len(animals)

  avg_animals = float(total_animals)/num_titles
  print "Average animals", avg_animals

def avg_body_parts(titles):
  total = 0
  num_titles = len(titles)
  for title in titles:
    nouns = libspacy.get_nouns(title)
    parts = [w for w in nouns if libwordnet.is_body_part(w) ]
    #if parts:
    #  print parts
    total +=len(parts)

  avg = float(total)/num_titles
  print "Average body_parts", avg

def avg_nes(titles):
  total = 0
  num_titles = len(titles)
  for title in titles:
    nes = libspacy.get_nes(title)
    #print nes
    total +=len(nes)

  avg = float(total)/num_titles
  print "Average NEs", avg


def avg_nes_halves(titles):
  total_f = total_s = total_b= 0
  num_titles = len(titles)
  for title in titles:
    nes = libspacy.get_nes(' '.join(title.split(' ')[:len(title.split(' '))/2]))
    #print nes
    na =int(len(nes)!=0)
    total_f +=int(len(nes)!=0)
    nes = libspacy.get_nes(' '.join(title.split(' ')[len(title.split(' '))/2:]))
    nb =int(len(nes)!=0)
    total_s +=int(len(nes)!=0)
    if na > 0 and nb > 0:
      total_b +=1

  avg_f = float(total_f)/num_titles
  avg_s = float(total_s)/num_titles
  print "Average NEs First half", avg_f, "total=", total_f
  print "Average NEs Second half", avg_s, "total=", total_s
  print "Total headlines with NE in both halves", total_b, "percent=", 1.0*total_b/num_titles

def get_top_verbs(titles):
  verbs_dict = {}
  for title in titles:
    verbs = libspacy.get_verbs(title)
    for verb in verbs:
      verb = str(verb)
      verbs_dict[verb] =verbs_dict.get(verb,0) + 1

  return sorted(verbs_dict.items(), key=lambda x:x[1], reverse=True)[:100]

def print_len_distributions(titles):
  lengths=[]
  num_titles = len(titles)
  for title in titles:
    lengths.append(len(title.split(' ')))

  for length in range(1,21):
    num_docs = sum([1 if l==length else 0 for l in lengths])
    percent = 100.0*num_docs/num_titles
    print "Length=%d num_docs=%d percentage=%f" % (length, num_docs, percent)

def nvn_phrases(titles):
  total = 0
  num_titles = len(titles)
  for title in titles:
    nvps = libspacy.get_noun_verb_pos(title)
    if 'NVN' in nvps:
      total += 1

  avg = float(total)/num_titles
  print "Average NVNs ", avg

def load_countries():
  fp = open('countries.txt', 'r')
  for line in fp:
    countries.append(line.strip().lower())
  #print countries
  return countries

def print_num_articles_with_popular_subject(titles, subjects):
  count = 0
  num_titles = len(titles)
  for title in titles:
    words = title.lower().split(' ')
    if sum([1 if word in subjects else 0 for word in words]) > 0:
      count +=1
  print "Number of articles containing most important subjects", count, "percent=", 1.0*count/num_titles

if __name__ == "__main__":
  load_countries()
  #title='Russian parliament grants Putin right to use military force in Syria'
  #print generate_features(title)

  main()
