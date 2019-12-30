from flask import Flask, render_template, request
from pymongo import MongoClient
from pprint import pprint

import configparser
import database
# import plot


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


import json

with open('data.json') as f:
  data = json.load(f)

with open('data_today.json') as f:
  data1=json.load(f)

database.connect_mongo(host,'traffic_data','alpha')

app = Flask(__name__)

weekdays=[]
count=[]
@app.route('/')
def index():
 return render_template('home.html')

# @app.route('/plot.png')
# def plot1():
#   weekdays, counts = database.get_from_mongo(90)
#   return plot.plot_png(weekdays, counts)
data_to_html=[]
@app.route('/chart',methods=['GET', 'POST'])
def chart():
  # print(request.is_xhr)
  
  if request.method == 'POST':
    value= request.form['ABC']
    value=int(value)
    if value!=1:
      data = database.get_from_mongo(value)
    else:
      data = database.get_from_mongo_today()
    
  else:
    data = database.get_from_mongo_today()

  if not request.is_xhr:
    return render_template("chart.html",data=data )

  else:
    data=json.dumps(data)
    return data
  
  


@app.route('/alpha')
def index_alpha():
  b=0
  for x in range(len(data1)+1):
    if x%100==0 and x!=0:
      a=b
      b=x
      mongo_docs=data1[a:b]
      database.write_to_mongo(mongo_docs)
      time.sleep(2)
  return render_template('alpha.html')


if __name__ == '__main__':
  app.run(debug=True)