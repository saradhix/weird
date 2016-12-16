import pymongo
import sys
import json

conn = pymongo.MongoClient()
db = conn.test
coll = db.articles_new

sources=['www.thehindu.com', 'www.nytimes.com', 'www.businessinsider.com',
        'bbc.co.uk', 'www.foxnews.com', 'www.deccanchronicle.com',
        'www.ibtimes.co.in', 'www.latimes.com', 'www.usatoday.com',
        'www.news18.com', 'www.rediff.com', 'www.nbcnews.com',
        'www.ibnlive.com'
        ]
for src in sources:
  docs = coll.find({'src':src})
  num_docs=0
  for doc in docs:
    json_obj={'title':doc['tit'].encode('utf-8'), 'url':doc['url'].encode('utf-8')}
    print json.dumps(json_obj)
    num_docs +=1
#  print "num_docs=", num_docs
