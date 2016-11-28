from spacy.parts_of_speech import ADV, ADJ, VERB, NOUN
from spacy.en import English
import spacy


nlp = English()
#probs = [lex.prob for lex in nlp.vocab]
#probs.sort()
def get_adjectives(sentence):
  adjs = set()
  sentence = nlp(sentence.decode('utf-8'))
  for token in sentence:
    if token.pos == spacy.parts_of_speech.ADJ:
      adjs.add(token.string)
  return adjs

#Gives a tuple of counts in this sequence Noun, Verb, Adj, Adv
def get_pos_counts(sentence):
  pos_counts=[0,0,0,0]
  sentence = nlp(sentence.decode('utf-8'))
  for token in sentence:
    if token.pos == spacy.parts_of_speech.NOUN:
      pos_counts[0] +=1
    if token.pos == spacy.parts_of_speech.VERB:
      pos_counts[1] +=1
    if token.pos == spacy.parts_of_speech.ADJ:
      pos_counts[2] +=1
    if token.pos == spacy.parts_of_speech.ADV:
      pos_counts[3] +=1
  return pos_counts
'''
s = "A healthy king lives happily"
print get_adjectives(s)
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
