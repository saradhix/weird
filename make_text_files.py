import json
import sys
def main():
  weird_news='weird.json'
  normal_news='normal2.json'

  raw_weird=[]
  raw_normal=[]
  #Load weird news
  fd = open(weird_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_weird.append(str(title))

  fd.close()
  print( "Weird news items :", len(raw_weird))

  #Load normal news
  fd = open(normal_news, 'r')
  for line in fd:
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])
    raw_normal.append(str(title))
  fd.close()

  print( "Normal news items :", len(raw_normal))

  fd =open('weird.txt','w')
  for title in raw_weird:
    fd.write(title+'\n')
  fd.close()
  fd =open('normal.txt','w')
  for title in raw_normal:
    fd.write(title+'\n')
  fd.close()

if __name__ == "__main__":
  main()
