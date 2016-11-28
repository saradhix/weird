from bs4 import BeautifulSoup
import requests
import json


start_page=1
end_page=779

baseurl='http://in.reuters.com'


for page in range(start_page, end_page+1):

  url='http://in.reuters.com/news/archive/oddlyEnoughNews?view=page&page='+str(page)+'&pageSize=10'


  r=requests.get(url)
  data=r.text
  soup=BeautifulSoup(data,"html.parser")

  for link in soup.find_all("h3", { "class" : "story-title" }):
    title = link.contents[0].text.encode('utf8')
    url = baseurl+link.contents[0].get('href').encode('utf8')
    #print title,',',url

    jsonobj={'title':title,'url':url}
    print json.dumps(jsonobj)
