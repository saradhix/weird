import json

filename = 'output'

with open(filename) as fp:
  for line in fp:
    line = ''.join([i if ord(i) < 128 else ' ' for i in line.strip()])
    article_id = (line[0:8]).strip()
    article = line[8:]
    json_obj = json.loads(article)
    lang = json_obj.get('lang','U')
    if lang == 'en':
      #print json_obj['tit']
      new_json = {'title':json_obj['tit'], 'url':json_obj['url']}
      print json.dumps(new_json)

