import json 
def main():
  upi_data='upi.json'
  fd = open(upi_data, 'r')
  for line in fd:
    if "null" in line: continue
    json_obj = json.loads(line)
    title = ''.join([i if ord(i) < 128 else ' ' for i in json_obj['title']])

    if len(title.split(' ')) <=3:
      continue
    print line.strip()



if __name__ == "__main__":
  main()
