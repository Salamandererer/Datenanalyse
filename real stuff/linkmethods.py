import pandas as pd
import psutil
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from urllib import parse
from bs4 import BeautifulSoup
from collections import Counter
import numpy as np
import os

URL = "https://wikimedia.org/api/rest_v1/"
HEADERS = {"Accept": "application/json", "user-agent": "marvin.braun1@smail.inf.h-brs.de"}


def get_pageviews(article: str,
                  start = datetime(2015, 7, 1),
                  end = datetime(2022, 7, 1),
                  project="de.wikipedia.org", access="all-access", agent="all-agents", granularity="daily"):
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
    results = []
    try:
        results = requests.get(url, headers=HEADERS).json()["items"]
        return results
    except:
        pass


def get_views(article: str):
    data = get_pageviews(article, project="de.wikipedia.org")
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
        "gpllimit": "max",
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
        "bllimit": "max",
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


############################################################################################################
if __name__ == '__main__':

    views = []
    timestamp = []
    year = []

    # get data from one article
    data = get_pageviews("Meisen", project="de.wikipedia.org")

    # Get the timestamp of the first set of Data
    firstyear = [int(data[0]["timestamp"][0:4])]
    for entry in data:
        article = entry['article']
        timestamp.append((entry['timestamp']))
        views.append((entry['views']))

    x = np.linspace(firstyear, 2022, num=len(views))
    plt.plot(x, views)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article)
    plt.show()

    dataM = get_forward_links("Meisen")
    dataB = get_back_links("Meisen")
    print("joo bin da")
    print(len(dataB))
    new2 = [k for k in dataB if not 'Benutzer' in k]
    print(len(new2))
    print(new2)
    new3 = [k for k in new2 if not 'Diskussion' in k]
    new4 = [k for k in new3 if not 'Wikipedia' in k]
    new5 = [k for k in new4 if not '/' in k]

    path = "datafiles/Meisen"
    print(path)

    for entry in new5:
        df = pd.DataFrame(get_pageviews(entry))
        df.to_csv(path + '/' f'{entry}' + '.csv', index=False)
        print(df)

    '''
    filterdf = pd.DataFrame(dataB)
    filterdf.columns = ["Backlinks"]
    filterdf2 = filterdf[filterdf["Backlinks"].str.contains("Benutzer")]
    filterdf3 = filterdf[filterdf["Backlinks"].str.contains("Diskussion")]
    print(filterdf2)
    print(filterdf3)
    print("b4",len(filterdf))
    print(filterdf)
    print("after", len(filterdf))
    print(len(filterdf)-len(filterdf2))
    '''

# Can remove unusefull links via:
# dataB.remove("Magdeburger Straßen/M")
# dataB.remove("Benutzer Diskussion:Nina/Archiv2")
# dataB.remove("Benutzer:Atamari/Liste der Vögel in Gambia")
