def make_bigrams(text):
  words_a = text.split(' ')
  words_b = words_a[1:]
  return zip(words_a, words_b)

def make_trigrams(text):
  words_a = text.split(' ')
  words_b = words_a[1:]
  words_c = words_a[2:]
  return zip(words_a, words_b, words_c)

def make_4grams(text):
  words_a = text.split(' ')
  words_b = words_a[1:]
  words_c = words_a[2:]
  words_d = words_a[3:]
  return zip(words_a, words_b, words_c, words_d)
def ngram(sentence,n): 
  input_list = [elem for elem in sentence.split(" ") if elem != '']
  return zip(*[input_list[i:] for i in xrange(n)])
#a='one two three four five'
#print make_trigrams(a)
