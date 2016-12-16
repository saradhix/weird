from spacy.parts_of_speech import ADV, ADJ, VERB, NOUN
from spacy.en import English
import spacy


nlp = English()
#probs = [lex.prob for lex in nlp.vocab]
#probs.sort()
def get_nouns(sentence):
  nouns = set()
  #sentence = nlp(str(sentence.decode('utf-8'))
  sentence = nlp(sentence)
  for token in sentence:
    if token.pos == spacy.parts_of_speech.NOUN:
      nouns.add(token.string)
  return list(nouns)

def get_verbs(sentence):
  verbs = set()
  #sentence = nlp(sentence.decode('utf-8'))
  sentence = nlp(sentence)
  for token in sentence:
    if token.pos == spacy.parts_of_speech.VERB:
      verbs.add(token.string)
  return list(verbs)

def get_nes(sentence):
  #parsed=nlp(sentence.decode('utf-8'))
  parsed=nlp(sentence)
  nes = [ i.label_ for i in parsed.ents]
  return nes

#Gives a tuple of counts in this sequence Noun, Verb, Adj, Adv
def get_pos_counts(sentence):
  pos_counts=[0 for i in range(7)]
  #sentence = nlp(sentence.decode('utf-8'))
  sentence = nlp(sentence)
  for token in sentence:
    if token.pos == spacy.parts_of_speech.NOUN:
      pos_counts[0] +=1
    if token.pos == spacy.parts_of_speech.VERB:
      pos_counts[1] +=1
    if token.pos == spacy.parts_of_speech.ADJ:
      pos_counts[2] +=1
    if token.pos == spacy.parts_of_speech.ADV:
      pos_counts[3] +=1
    if token.pos == spacy.parts_of_speech.DET:
      pos_counts[4] +=1
    if token.pos == spacy.parts_of_speech.PUNCT:
      pos_counts[5] +=1
    if token.pos == spacy.parts_of_speech.CONJ:
      pos_counts[6] +=1
  return pos_counts

def get_noun_verb_pos(sentence):
  ret=[]
  sentence = nlp(sentence)
  for token in sentence:
    if token.pos == spacy.parts_of_speech.NOUN:
      ret.append('N')
    if token.pos == spacy.parts_of_speech.VERB:
      ret.append('V')

  return ''.join(ret)

def get_nsubj(sentence):
  parsed = nlp(sentence)
  return [ i for i in parsed if i.dep_ == "nsubj"]
'''
s = "A healthy king lives happily"
print get_nsubj(s)
s = "I am very rich and beautiful girl"
print get_adjectives(s)
'''
'''
sentence = nlp(u'A healthy man lives happily')
print sentence
for token in sentence:
  print token, token.pos, is_adverb(token)
'''

'''
s='A happy dog barks happily'
print get_pos_counts(s)
'''
