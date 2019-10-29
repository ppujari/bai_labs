from flask import Flask, render_template
from pymongo import MongoClient
from pprint import pprint

import configparser
import database


from datetime import datetime
from datetime import date
now = datetime. now()
timestamp = datetime.timestamp(now)


parser = configparser.ConfigParser()
parser.read('./config/config.ini')


import time
host = (parser.get('DEV', 'DATABASE'))



from datetime import date, timedelta
import random
import calendar

# Assuming you want a random day of the current year


# print(randomDay)

import json

with open('data.json') as f:
  data = json.load(f)

database.connect_mongo(host,'traffic_data','alpha')

app = Flask(__name__)


@app.route('/')
def index():
 return render_template('home.html')

@app.route('/alpha')
def index_alpha():
  b=0
  for x in range(len(data)+1):
    if x%100==0 and x!=0:
      a=b
      b=x
      print(a,b)
      # data1=[{"store_id": 100, "time_stamp": "2019-10-26 22:34:45.876376", "people_count": 35, "gender": "F", "age_group": "01-20"},
      # {"store_id": 100, "time_stamp": "2019-10-26 22:34:45.876376", "people_count": 35, "gender": "F", "age_group": "01-20"}]
      mongo_docs=data[a:b]
      # mongo_docs=data1
      database.write_to_mongo(mongo_docs)
      time.sleep(2)
  return render_template('alpha.html')

if __name__ == '__main__':
  app.run(debug=True)