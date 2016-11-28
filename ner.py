import spacy.en
nlp = spacy.en.English()

def get_nes(sentence):
    parsed=nlp(sentence)
    print parsed.ents
    for entity in parsed.ents:
        print entity
        print  entity.label_, ' '.join(t.orth_ for t in entity)

sentence = u"PM Modi is on a tour to China and America and California"
get_nes(sentence)
