import json
filename='veooz_10k.txt'
ft=open('normal4.json','w')
with open(filename) as fp:
  for line in fp:
    title = line.strip()
    json_obj={'url':None,'title':title}
    json_str = json.dumps(json_obj)
    ft.write(json_str+'\n')

ft.close()
