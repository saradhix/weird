import json 
ignore_words = ['review','commentary', 'jockstrip', 'quirky', 'interview', 'quirks']
def main():
  upi_data='upi.json'
  fd = open(upi_data, 'r')
  for line in fd:
    if "null" in line: continue
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])

    #Replace :, . and ,
    title = title.replace(':','')
    title = title.replace(',','')
    title = title.replace('.','')
    title = title.replace('?','')

    if len(title.split(' ')) <=3:
      continue
    words = title.split(' ')
    lower_words = set([ i.lower() for i in words])
    ignore = set(ignore_words)

    if len(list(lower_words & ignore)):
      continue
    
    print line.strip()



if __name__ == "__main__":
  main()
