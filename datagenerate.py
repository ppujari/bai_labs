import json
import random 
# impo
# data = {}

from datetime import datetime, date, timedelta

import calendar


now = datetime.now()
# print(now)
# current_time = now.strftime("%H:%M:%S")
# print(current_time)
firstJan = date.today().replace(day=1, month=1) 

# randomDay = firstJan + timedelta(days = random.randint(0, 365 if calendar.isleap(firstJan.year) else 364))


data= []

for i in range(100):
  now = datetime.now()
  # print(now)
  data.append({
      "store_id":random.choice([100]),
      # "time_stamp": str(now),
      # "time_stamp": str(firstJan + timedelta(days = random.randint(0, 365 if calendar.isleap(firstJan.year) else 364))),
      "date": str(date.today() ),
      "time_stamp": str(random.randrange(0, 24)),
      "people_count":i,
      "gender": random.choice(['M', 'F']),
      "age_group":random.choice(['01-20', '21-40', '41-55', '55-70', '70-99'])
  })



with open('data_today.json', 'w') as outfile:
    json.dump(data, outfile)