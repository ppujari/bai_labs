from pymongo import MongoClient

col = ""
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