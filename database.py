from pymongo import MongoClient
from datetime import date, timedelta, datetime
import json
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

    except Exception as e:
        print(e) 
        connect=0
        
    if connect==1:
        print("connected")
    return connect



def get_from_mongo(value):
    global col
    global days
    total_count=[]
    mcounts=[]
    fcounts=[]

    pipeline=[{'$group': { '_id': '$time_stamp', 'mcount':{"$sum": { "$cond": [{ "$eq": ["M","$gender"] },1,0]}},
    'fcount':{"$sum": { "$cond": [{ "$eq": ["F","$gender"] },1,0]}}
    }},{'$sort':{'_id': 1}}]
    values= list(col.aggregate(pipeline))
    
    
    filter(value)

    np=False
    for i, y in enumerate(days):
        if not np and i != 0:
            mcounts.append(0)
            fcounts.append(0)
        np=False   
        for x in range(len(values)):
            var2=values[x]
            var1=var2['_id']
            if(var1==y):
                np=True
                mcounts.append(var2['mcount'])
                fcounts.append(var2['fcount'])
          

    for i in range(0, len(mcounts)): 
        total_count.append(mcounts[i] + fcounts[i])

    yma=movingaverage(total_count,2)

    data={
        "x": days,
        "m": mcounts,
        "f": fcounts,
        "a":yma
    }
    
    return data



def get_from_mongo_today():
    global col
    pipeline=[{'$match': {'date': '2019-12-28'}},{'$group': { '_id': '$time_stamp', 'mcount':{"$sum": { "$cond": [{ "$eq": ["M","$gender"] },1,0]}},
    'fcount':{"$sum": { "$cond": [{ "$eq": ["F","$gender"] },1,0]}}
    }},{'$sort':{'_id': 1}}]

    values= list(col.aggregate(pipeline))
    
    now = datetime.now() 

    total_count=[]
    mcounts=[]
    fcounts=[]
    dayss=[]
    filter(1)

    for y in range(24):
        for x in range(len(values)):
            var2=values[x]
            var1=var2['_id']
            if(int(var1)==y):
                dayss.append(int(var1))
                if y <= now.hour:
                    mcounts.append(var2['mcount'])
                    fcounts.append(var2['fcount'])
                else:
                    mcounts.append(0)
                    fcounts.append(0)

    for i in range(0, len(mcounts)): 
        total_count.append(mcounts[i] + fcounts[i])

    yma=movingaverage(total_count,2)

    data={
        "x": dayss,
        "m": mcounts,
        "f": fcounts,
        "a": yma
    }
    return data
    

def movingaverage (values, window=2):
    df = pd.Series(values).rolling(2).mean().fillna(0)
    df1=df.to_numpy()
    df2=list(df1)
    return list(df2)

def filter(fil):
    global days
    days=[]
    now = datetime.now()

    for x in range(fil):
        days.append((now-timedelta(fil-x-1)).strftime("%Y-%m-%d"))


def write_to_mongo(mongo_docs):
    global col
    batch_count = 100
    count=len(mongo_docs)

    try :
        if len(mongo_docs) <= batch_count:
            try:
                col.insert_many(mongo_docs)
                mongo_docs = []
               
            except Exception as e:
                print(e)
                pass
        print ("Inserted row count: " ,count)
    except Exception as e:
        print(e)
        pass