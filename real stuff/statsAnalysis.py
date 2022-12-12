import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from urllib import parse

URL = "https://wikimedia.org/api/rest_v1/"
HEADERS = {"Accept": "application/json", "user-agent": "marvin.braun1@smail.inf.h-brs.de"}


def get_pageviews(article: str, start: datetime, end: datetime, project="de.wikipedia.org",
                  access="all-access", agent="all-agents", granularity="daily"):
    """
        args:
            article: The name of the article
            start: Start date
            end: End date
            project: The domain, default en.wikipedia.org
            access: Type of the device, default all-access. other options(Desktop, mobile-app, mobile-web)
            agent: Type of the agent, default all-agents. other options(user, spider, automated)
            granularity: The time unit, default daily. Other options(monthly)

    """
    params = [
        "metrics",
        "pageviews",
        "per-article",
        project.capitalize(),
        access,
        agent,
        parse.quote(article),
        granularity,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d")
    ]
    url = URL + "/".join(params)
    return requests.get(url, headers=HEADERS).json()["items"]

def analysis(data):
    views = []
    timestamp = []

    for entry in data:
        article = entry['article']
        timestamp.append((entry['timestamp']))
        views.append((entry['views']))

    return data


####################################################################################
syear = 2015
start = datetime(syear, 1, 1)
end = datetime.today()
article = "Bundestagswahl 2021"
#dict = {'project': 'de.wikipedia.org', 'article': 'Meisen', 'granularity': 'daily', 'timestamp': '2022101100',
        #'access': 'all-access', 'agent': 'all-agents', 'views': 121}
# print(dict3[0]['article'])

#data = get_pageviews(article, start, end, "de.wikipedia.org", "all-access", "all-agents", "daily")
#get data from one article

data = get_pageviews(article, start, end, project="de.wikipedia.org")
#print(data[0]['timestamp'])
views = []
timestamp = []
year = []
zero = 0

firstyear = [int(data[0]["timestamp"][0:4])]
print(firstyear)

for entry in data:
    article = entry['article']
    timestamp.append((entry['timestamp']))
    views.append((entry['views']))

#for entry in data:
#Timecorrection
#c = (len(views)// 365)
#crest = len(views) % 365
#print(c)
#for i in range(c):
#   year.append(syear + i)
#print(year)

#lastTimestamp = views.pop(entry['timestamp'])
#lastYear = lastTimestamp[0:3]
#lastMonth = lastTimestamp[4:5]
#lastTime = int (lastYear)

#x = np.linspace(firstyear, 2022.5, num=len(views))
#plt.plot(x, views)
#plt.xlabel("Time")
#plt.ylabel("Views")
#plt.title(article)
#plt.show()
# for i in views:
#    print((views[i]))
#   print((timestamp))
#   print((article))


#######
syear2 = 2015
start2 = datetime(syear2, 1, 1)
end2 = datetime.today()
article2 = "Bundesregierung"

data2 = get_pageviews(article2, start2, end2, project="de.wikipedia.org")

views2 = []
timestamp2 = []
year2 = []

firstyear2 = []
firstyear2.append(int(data[0]["timestamp"][0:4]))

for entry in data2:
    article2 = entry['article']
    timestamp2.append((entry['timestamp']))
    views2.append((entry['views']))


#x2 = np.linspace(firstyear2, 2022, num=len(views2))
#plt.plot(x2, views2)
#plt.xlabel("Time")
#plt.ylabel("Views")
#plt.title(article2)
#plt.show()

#########
