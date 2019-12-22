from pymongo import MongoClient
from datetime import date, timedelta
import json
from bunch import bunchify
from datetime import datetime

col = ""
days=[]
def connect_mongo(ip, collection='traffic_data', database='alpha'):
    global col
    connect=1
    try:    
        print ("Connecting to mongodb ..")
        client = MongoClient(ip)
        db= client[database]
        col= db[collection]
        print(col)
    except Exception as e:
        print(e) 
        connect=0
        
    if connect==1:
        print("connected")
    return connect

def get_from_mongo(value):
    global col
    global days
    # print(col)
    pipeline=[{'$group': { '_id': '$time_stamp', 'count':{'$sum':1}}},{'$sort':{'_id': 1}}]
    # values= list(col.aggregate(pipeline))
    values= list(col.aggregate(pipeline))
    # count = col.count_documents({})
    
    # print(type(values))
    # print(values)

    
    now = datetime.now() # current date and time
    # weekday= now.weekday()
    # weekdays=[]
    # weekdays.append((now-timedelta(weekday)).strftime("%Y/%m/%d"))
    date_time = now.strftime("%Y/%m/%d")
    print("date and time:",date_time)
    
    # print(weekday)


    # for x in range(7):
    #     if(x<weekday):
    #         weekdays.append((now-timedelta(weekday-x)).strftime("%Y-%m-%d"))
    #     elif(x==weekday):
    #          weekdays.append((now-timedelta(0)).strftime("%Y-%m-%d"))
    #     else:
    #          weekdays.append((now+timedelta(x-weekday)).strftime("%Y-%m-%d"))

    # for x in range(7):
    #     weekdays.append((now-timedelta(6-x)).strftime("%Y-%m-%d"))
    # print(weekdays)  
    # print(len(values))
    counts=[]
    filter(value)
    # print('amiya')
    for x in range(len(values)):
        beb=values[x]
        ami=beb['_id']
        # print(ami)
        for y in days:
            if(ami==y):
                counts.append(beb['count'])
    # print(weekdays)
    # print(counts)

    data={
        "x": days,
        "y": counts
    }
    print("data")

    return data


def filter(fil):
    global days
    days=[]
    now = datetime.now()
    
    for x in range(fil):
        days.append((now-timedelta(fil-x-1)).strftime("%Y-%m-%d"))
        
    print(days)

def write_to_mongo(mongo_docs):

    global col
    batch_count = 100
    count=len(mongo_docs)

    try :
        if len(mongo_docs) <= batch_count:
            try:
                # print(mongo_docs)
                res = col.insert_many(mongo_docs)
                mongo_docs = []
                print(res)
            except Exception as e:
                print(e)
                pass
        print ("Inserted row count: " ,count)
    except Exception as e:
        print(e)
        pass