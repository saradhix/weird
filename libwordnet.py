import textblob


def traverse(word, category):
  if word is None:
    return False
  #print word
  #print word.lemma_names()
  if category in word.lemma_names():
    return True

  hypernyms = word.hypernyms()
  if hypernyms:
    word = hypernyms[0]
    return traverse(word,category)
  return False


def is_animal(word):
  #print "Entered is_animal with ", word
  w = textblob.Word(word).synsets
  if len(w)==0:
    return False
  ss = w[0]
  return traverse(ss, "animal")

def is_body_part(word):
  #print "Entered is_animal with ", word
  w = textblob.Word(word).synsets
  if len(w)==0:
    return False
  ss = w[0]
  return traverse(ss, "body_part")
'''
words=['cat','snake','cup', 'book', 'dog', 'hare', 'board', 'grass', 'vulture', 'pig']

for word in words:
  print word, is_animal(word)
'''
