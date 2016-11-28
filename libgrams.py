def make_bigrams(text):
  words_a = text.split(' ')
  words_b = words_a[1:]
  return zip(words_a, words_b)

def make_trigrams(text):
  words_a = text.split(' ')
  words_b = words_a[1:]
  words_c = words_a[2:]
  return zip(words_a, words_b, words_c)

#a='one two three four five'
#print make_trigrams(a)
