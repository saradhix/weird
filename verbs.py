from spacy.symbols import nsubj, VERB
# Finding a verb with a subject from below --good
import spacy
nlp = spacy.load('en')
def get_verbs(doc):
  doc = nlp(doc.decode('utf-8'))
  verbs = []
  ents=[]
  for possible_subject in doc:
    if possible_subject.dep == nsubj and possible_subject.head.pos == VERB:
      verbs.append(possible_subject.head)
  for ent in doc.ents:
    print ent.label_, ent.text

  return list(set(verbs))

def get_nsubj(sentence):
  parsed = nlp(sentence.decode('utf-8'))
  return [ i for i in parsed if i.dep_ == "nsubj"]


def get_pobj(sentence):
  parsed = nlp(sentence.decode('utf-8'))
  return [ i for i in parsed if i.dep_ == "pobj"]

def get_nouns(sentence):
  parsed = nlp(sentence.decode('utf-8'))
  return [ i for i in parsed if i.pos == spacy.parts_of_speech.NOUN]

def get_noun_chunks(sentence):
  parsed = nlp(sentence.decode('utf-8'))
  return [ i.text for i in parsed.noun_chunks]

with open('weird/weird.txt') as fp:
  for line in fp:
    line = line.strip()
    print line
    #print get_verbs(line)
    #print get_nsubj(line), get_pobj(line) 
    print get_noun_chunks(line)
