import json
import sys
import nltk
from nltk.corpus import stopwords
import libspacy

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
    raw_weird.append(title)

  print "Weird news items :", len(raw_weird)

  #Load normal news
  fd = open(normal_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_normal.append(title)

  print "Normal news items :", len(raw_normal)

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

if __name__ == "__main__":
  main()
