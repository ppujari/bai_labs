from flask import Flask, render_template, request
from pymongo import MongoClient
from pprint import pprint
import configparser
import database
import time
import random
import calendar
import json


weekdays=[]
count=[]
data_to_html=[]


# **************Get the DataBase Host Details*****************
parser = configparser.ConfigParser()
parser.read('./config/config.ini')
host = (parser.get('DEV', 'DATABASE'))


# ****************** Load Json Data for inserting to database ******************
with open('data.json') as f:
  data = json.load(f)

with open('data_today.json') as f:
  data1=json.load(f)

database.connect_mongo(host,'traffic_data','alpha')

app = Flask(__name__)

#  ************************ HOME ROUTE ****************************
@app.route('/')
def index():
 return render_template('home.html')

# ************************* PLOTING GRAPH **********************

@app.route('/chart',methods=['GET', 'POST'])
def chart():
  
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
  

# ***************** INSERT INTO DATABASE ******************
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