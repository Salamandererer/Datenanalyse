import numpy as np
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from urllib import parse
from linkmethods import get_pageviews

URL = "https://wikimedia.org/api/rest_v1/"
HEADERS = {"Accept": "application/json", "user-agent": "marvin.braun1@smail.inf.h-brs.de"}

def analysis(data):
    views = []
    timestamp = []
    article = []

    for entry in data:
        article = entry['article']
        timestamp.append((entry['timestamp']))
        views.append((entry['views']))

    return data


####################################################################################
if __name__ == '__main__':
    views = []
    timestamp = []
    year = []
    article = "Bundestagswahl 2021"

    # dict = {'project': 'de.wikipedia.org', 'article': 'Meisen', 'granularity': 'daily', 'timestamp': '2022101100',
    # 'access': 'all-access', 'agent': 'all-agents', 'views': 121}
    # print(dict[0]['article'])

    # get data from one article
    data = get_pageviews(article, project="de.wikipedia.org")
    zero = 0

    firstyear = [int(data[0]["timestamp"][0:4])]

    for entry in data:
        article = entry['article']
        timestamp.append((entry['timestamp']))
        views.append((entry['views']))

    x2 = np.linspace(firstyear, 2022.5, num=len(views))
    plt.plot(x2, views)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article)
    plt.show()

    #######
    article2 = "Bundesregierung"
    data2 = get_pageviews(article2, project="de.wikipedia.org")

    views2 = []
    timestamp2 = []
    year2 = []
    firstyear2 = []
    firstyear2.append(int(data[0]["timestamp"][0:4]))

    for entry in data2:
        article2 = entry['article']
        timestamp2.append((entry['timestamp']))
        views2.append((entry['views']))

    x2 = np.linspace(firstyear2, 2022.5, num=len(views2))
    plt.plot(x2, views2)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article2)
    plt.show()

    #########
