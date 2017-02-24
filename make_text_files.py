import json
import sys
def main():
  weird_news='weird.json'
  normal_news='normal3.json'

  raw_weird=[]
  raw_normal=[]
  weird_urls=[]
  normal_urls=[]
  #Load weird news
  fd = open(weird_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_weird.append(str(title))
    weird_urls.append(json_obj['url'])

  fd.close()
  print( "Weird news items :", len(raw_weird))

  #Load normal news
  fd = open(normal_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_normal.append(str(title))
    normal_urls.append(json_obj['url'])
  fd.close()

  #Load stopwords
  stopwords=[]
  fd=open('stopwords','r')
  for stopword in fd:
    stopword=stopword.strip()
    if stopword:
      stopwords.append(stopword.strip())
  fd.close()
  print len(stopwords)
  #print stopwords
  print( "Normal news items :", len(raw_normal))
  words={}
  fd =open('weird.txt','w')
  for title in raw_weird:
    fd.write(title+'\n')
    for word in title.split(' '):
      word = word.lower()
      if word in stopwords:
        continue
      if len(word)<=2:
        continue
      words[word]=words.get(word,0)+1

  fd.close()
  fd =open('weird_urls.txt','w')
  for url in weird_urls:
    fd.write(url+'\n')
  fd.close()
  fd =open('normal_urls.txt','w')
  for url in normal_urls:
    fd.write(url+'\n')
  fd.close()

  s=sorted(words.items(), key=lambda x:x[1], reverse=True)
  for k in s:
    #print k[0], k[1]
    pass


  words={}
  fd =open('normal.txt','w')
  for title in raw_normal:
    fd.write(title+'\n')
    for word in title.split(' '):
      word = word.lower()
      if word in stopwords:
        continue
      if len(word)<=2:
        continue
      words[word]=words.get(word,0)+1

  fd.close()
  s=sorted(words.items(), key=lambda x:x[1], reverse=True)
  for k in s:
    print k[0], k[1]

if __name__ == "__main__":
  main()
