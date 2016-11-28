import json
import sys
import nltk
from nltk.corpus import stopwords
import libspacy
import libgrams
import libwordnet

def main():
  print "Hello"

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

  '''
  print_sentence_structure(raw_weird)
  print_sentence_structure(raw_normal)
  print "-"*40
  print_num_stop_words(raw_weird)
  print_num_stop_words(raw_normal)
  print "-"*40
  print_avg_word_len(raw_weird)
  print_avg_word_len(raw_normal)
  print "-"*40
  #print_pos_distributions(raw_weird)
  #print_pos_distributions(raw_normal)
  #print "-"*40
  print_quoted_counts(raw_weird)
  print_quoted_counts(raw_normal)
  print "-"*40
  '''
  most_repeated_bigrams(raw_weird)
  most_repeated_bigrams(raw_normal)
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



def print_sentence_structure(titles):
  total_words = 0
  num_titles = len(titles)
  for title in titles:
    words = title.split(' ')
    total_words = total_words + len(words)

  words_per_title = float(total_words) / num_titles
  print "Words per title=", words_per_title


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
  total_counts = [0,0,0,0]
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

  avg_num_quotes = float(total_num_quotes)/num_titles
  print "Quoted chars", avg_num_quotes


def most_repeated_bigrams(titles):
  bigrams={}

  for title in titles:
   grams = libgrams.make_trigrams(title.lower())
   for gram in grams:
     bigrams[gram]=bigrams.get(gram,0)+1

  #for gram in bigrams:
    #print gram, bigrams[gram]
  #print "-"*40
  for tup in sorted(bigrams.items(), key=lambda x:x[1], reverse=True)[:20]:
    print tup
  #sys.exit()

def avg_capitalized_words(titles):
  total_caps = 0
  num_titles = len(titles)
  for title in titles:
    for word in title.split(' '):
      if word.upper()==word and word.lower() !=word:
        #print word
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

if __name__ == "__main__":
  main()
