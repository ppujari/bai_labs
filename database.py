from pymongo import MongoClient
from datetime import date, timedelta
import json
from bunch import bunchify
from datetime import datetime
import numpy as np
import pandas as pd


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
    # pipeline=[{'$group': { '_id': '$time_stamp', 'count':{'$sum':1}}},{'$sort':{'_id': 1}}]
    pipeline=[{'$group': { '_id': '$time_stamp', 'mcount':{"$sum": { "$cond": [{ "$eq": ["M","$gender"] },1,0]}},
    'fcount':{"$sum": { "$cond": [{ "$eq": ["F","$gender"] },1,0]}}
    }},{'$sort':{'_id': 1}}]
    values= list(col.aggregate(pipeline))
    # print(values)
    
    now = datetime.now() 
    date_time = now.strftime("%Y/%m/%d")
    print("date and time:",date_time)
    total_count=[]
    mcounts=[]
    fcounts=[]
    filter(value)

    np=False
    for i, y in enumerate(days):
        if not np and i != 0:
            print("i am nothing")
            mcounts.append(0)
            fcounts.append(0)
        np=False   
        for x in range(len(values)):
            beb=values[x]
            ami=beb['_id']
            if(ami==y):
                np=True
                mcounts.append(beb['mcount'])
                fcounts.append(beb['fcount'])
          

    for i in range(0, len(mcounts)): 
        total_count.append(mcounts[i] + fcounts[i])
    yma=movingaverage(total_count,2)

    data={
        "x": days,
        "m": mcounts,
        "f": fcounts,
        "a":yma
    }
    # print(data)
    

    return data
def get_from_mongo_today():
    global col
    pipeline=[{'$match': {'date': '2019-12-28'}},{'$group': { '_id': '$time_stamp', 'mcount':{"$sum": { "$cond": [{ "$eq": ["M","$gender"] },1,0]}},
    'fcount':{"$sum": { "$cond": [{ "$eq": ["F","$gender"] },1,0]}}
    }},{'$sort':{'_id': 1}}]
    # pipeline2={'collation': {'numericOrdering':'true'}}
    # pipeline=[{'$match': {'date': '2019-12-28'}},{'$group': {'_id': '$time_stamp'}}]
    values= list(col.aggregate(pipeline))
    # print(list(col.find({'date': '2019-12-28'})))
    # print("amiyajjjjjjjjjjjjjjjjjj")
    # print(values)
    
    now = datetime.now() 
    # print(now.hour)
    # date_time = now.strftime("%Y/%m/%d")
    # print("date and time:",date_time)
    total_count=[]
    mcounts=[]
    fcounts=[]
    dayss=[]
    filter(1)

    for y in range(24):
        for x in range(len(values)):
            beb=values[x]
            ami=beb['_id']
            if(int(ami)==y):
                dayss.append(int(ami))
                if y <= now.hour:
                    mcounts.append(beb['mcount'])
                    fcounts.append(beb['fcount'])
                else:
                    mcounts.append(0)
                    fcounts.append(0)
            # else:
            #     mcounts.append(0)
            #     fcounts.append(0)
    for i in range(0, len(mcounts)): 
        total_count.append(mcounts[i] + fcounts[i])
    yma=movingaverage(total_count,2)
    # print(yma)

    data={
        "x": dayss,
        "m": mcounts,
        "f": fcounts,
        "a": yma
    }


    return data
    

def movingaverage (values, window=2):
    print(values)
    df = pd.Series(values).rolling(2).mean().fillna(0)
    df1=df.to_numpy()
    df2=list(df1)
    # df2=df2[2:]
    # for x in range(2):
    #     df2.append(0)

    return list(df2)

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