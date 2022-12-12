from pathlib import Path
import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from urllib import parse
from urllib.parse import urlparse
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
import os
import glob

soup = BeautifulSoup(requests.get("https://de.wikipedia.org/wiki/Meisen").text, "html.parser")

foundUrls = Counter([link["href"] for link in soup.find_all("a", href=lambda href: href and not href.startswith("#"))])
foundUrls = foundUrls.__contains__('/wiki/')

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


############################################################################################################

syear = 2016
start = datetime(syear, 1, 1)
views = []
timestamp = []
year = []
end = datetime(2022, 1, 1)

# get data from one article
data = get_pageviews("Meisen", start, end, project="de.wikipedia.org")
# print(type(data))

# Get the timestamp of the first set of Data
firstyear = [int(data[0]["timestamp"][0:4])]
for entry in data:
    article = entry['article']
    timestamp.append((entry['timestamp']))
    views.append((entry['views']))
# print(timestamp)

# for entry in data:
# Timecorrection
# c = divmod(len(views), 365)
# modyear = c[0]
# print(len(views))
# realyear = []
# for i in range(modyear):
# realyear.append(firstyear[0] + i)
# print(realyear)
x = np.linspace(firstyear, 2022, num=len(views))
plt.plot(x, views)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title(article)
plt.show()


def get_views(article: str):
    data = get_pageviews(article, start, end, project="de.wikipedia.org")
    views = []
    for entry in data:
        views.append((entry['views']))
    return views


def get_forward_links(article: str):
    S = requests.Session()
    URL = "https://de.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "titles": article,
        "gpllimit": "50",
        "format": "json",
        "generator": "links"

    }
    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    PAGES = DATA['query']['pages']
    titels = []

    for page in PAGES.values():
        if 'missing' not in page:
            titels.append(page['title'])

    return titels


def get_back_links(article: str):
    S = requests.Session()
    URL = "https://de.wikipedia.org/w/api.php"

    PARAMS = {
        "action": "query",
        "bltitle": article,
        "bllimit": "50",
        "format": "json",
        "list": "backlinks",
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    BACKLINKS = DATA["query"]["backlinks"]
    blinks = []

    for b in BACKLINKS:
        blinks.append(b["title"])

    return blinks


def count_links(article: str):
    soup = BeautifulSoup(requests.get(f"https://de.wikipedia.org/wiki/{article}").text, "html.parser")
    foundUrls = Counter(
        [link["href"] for link in soup.find_all("a", href=lambda href: href and not href.startswith("#"))])
    foundUrls = foundUrls.most_common()
    links = []

    for item in foundUrls:
        links.append("%s: %d" % (item[0], item[1]))
    return links


dataM = get_forward_links("Meisen")
dataB = get_back_links("Meisen")
#print(dataM)
#print(dataB)
dataB.remove("Magdeburger Straßen/M")
dataB.remove("Benutzer Diskussion:Nina/Archiv2")
dataB.remove("Benutzer:Atamari/Liste der Vögel in Gambia")
#print(dataB)

# print(get_pageviews("Meisen", start, end, project="de.wikipedia.org"))
# print("count links:",count_links("Meisen"))
# print("Forwardlinks", get_forward_links("Meisen"))
# forwardlinks = pd.DataFrame(get_forward_links("Meisen"))
# forwardlinks.to_csv('MeisenForwardlinks.csv',index=True,index_label='index')
# backlinks = pd.DataFrame(get_back_links("Meisen"))
# backlinks.to_csv('MeisenBacklinks.csv',index=True,index_label='index')
path = os.path.abspath('Csvs/frommeisen')
for entry in dataB:
    data = get_pageviews(f"{entry}", start, end, project="de.wikipedia.org")
    df = pd.DataFrame(data)
    df.to_csv(path + f'\{entry}.csv',index=False)
# print(get_pageviews(dataB, start, end, project="de.wikipedia.org"))
# get_pageviews("Magdeburger Straßen/M", start, end, project="de.wikipedia.org")