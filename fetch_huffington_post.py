from bs4 import BeautifulSoup
import requests
import json
import sys
from lxml import html

start_page=1
end_page=17

baseurl='http://in.reuters.com'
headers = {'Accept-Encoding': 'deflate'}

for page in range(start_page, end_page+1):
  #print "Downloading page ", page

  url='http://www.huffingtonpost.co.uk/news/weird-news/'+str(page)+'/'


  html_tree = html.parse(url)
  root = html_tree.getroot()
  links = root.xpath("//h3[@class='margin_bottom_5']")
  for link in links:
    url=link[0].get('href')
    title=link[0].text.strip()
    #print title, url
    jsonobj={'title':title,'url':url}
    print json.dumps(jsonobj)


