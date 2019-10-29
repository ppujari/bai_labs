import json
import random 
# data = {}

from datetime import datetime, date, timedelta

import calendar


now = datetime.now()

firstJan = date.today().replace(day=1, month=1) 

# randomDay = firstJan + timedelta(days = random.randint(0, 365 if calendar.isleap(firstJan.year) else 364))


data= []

for i in range(1000):
  now = datetime.now()
  print(now)
  data.append({
      "store_id":random.choice([100]),
      # "time_stamp": str(now),
      "time_stamp": str(firstJan + timedelta(days = random.randint(0, 365 if calendar.isleap(firstJan.year) else 364))),
      "people_count":i,
      "gender": random.choice(['M', 'F']),
      "age_group":random.choice(['01-20', '21-40', '41-55', '55-70', '70-99'])
  })



with open('data.json', 'w') as outfile:
    json.dump(data, outfile)