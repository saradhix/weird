from bs4 import BeautifulSoup
import requests
import json
import sys


start_page=1
end_page=189

baseurl='http://www.upi.com/Odd_News/2018/p'


for page in range(start_page, end_page+1):
  url = baseurl+str(page)
  #print "Trying ", url



  r=requests.get(url)
  data=r.text
  soup=BeautifulSoup(data,"html.parser")

  for item in soup.find_all("div", { "class" : "upi_item" }):
    link = item.contents[1].get('href')
    title = item.contents[1].get('title')
    #title = link.contents[0].text.encode('utf8')
    #link = baseurl+link.contents[0].get('href').encode('utf8')
    #print title,',',link

    jsonobj={'title':title,'url':link}
    print(json.dumps(jsonobj))
