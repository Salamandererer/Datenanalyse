import glob
import sys

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
                  start=datetime(2015, 7, 1),
                  end=datetime(2022, 7, 1),
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
        "plcontinue": True,
        "bllimit": "max",
        "format": "json",
        "list": "backlinks",
    }
    data = get_pageviews(article)
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


def get_target(target: str):
    try:
        write_backlinks_tocsv(target)
    except:
        pass
    path = r'c:\Users\Marvin\PycharmProjects\pythonProject/real stuff/datafiles/' + target + '/' + target + '.csv'
    data = pd.read_csv(path)
    df = pd.DataFrame(data)[["timestamp", "views"]]
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def get_backlink_views(target, ref_df):
    path = r'c:\Users\Marvin\PycharmProjects\pythonProject/real stuff/datafiles/' + target + '/backlinksdata'
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    flatten_views = []

    for f in csv_files:
        df = pd.read_csv(f)
        df = df[["timestamp", "views"]].set_index("timestamp").reindex_like(ref_df).fillna(0)
        print(df)

        views = df.views.to_numpy()
        flatten_views.append(views)
    return flatten_views


def get_backlink_strings(article):
    path = r'c:\Users\Marvin\PycharmProjects\pythonProject/real stuff/datafiles/' + article + '/backlinksdata'  # make path a global variable
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    cropped_result = [r[0:-4] for r in result]
    return cropped_result
def get_links(article: str, prop: str):

    "props = links or linkshere"

    if prop == "links":
        limit = "pllimit=max"
        con = "plcontinue="
    else:
        limit = "lhlimit=max"
        con = "lhcontinue="

    titles = []
    data = requests.get(f"https://de.wikipedia.org/w/api.php?action=query&format=json&titles={article}&prop={prop}&{limit}").json()
    data_continue = data

    data = to_links(data, prop)
    for i in data:
        titles.append(i["title"])

    while list(data_continue.keys())[0] == "continue":
        data_continue = data_continue["continue"]
        data_continue = data_continue[con[0:len(con)-1]]
        data_continue = requests.get(f"https://de.wikipedia.org/w/api.php?action=query&format=json&titles={article}&prop={prop}&{limit}&{con}{data_continue}").json()
        data_list = to_links(data_continue, prop)
        for i in data_list:
            titles.append(i["title"])

    return titles

def to_links(data: dict, prop: str):
    data = data["query"]
    data = data["pages"]
    data = data[list(data.keys())[0]]
    return data[f"{prop}"]


def write_backlinks_tocsv(target: str):
    if not isinstance(target, str):
        raise TypeError('Target has to be a string')
    else:
        data = get_links(target,"linkshere")
        new = [k for k in data if not 'Benutzer' in k]
        new1 = [k for k in new if not 'Diskussion' in k]
        new2 = [k for k in new1 if not 'Wikipedia' in k]
        new3 = [k for k in new2 if not '/' in k]
        pathindir = "datafiles/" + target
        pathofblinkdata = "datafiles/" + target + "/backlinksdata"
        os.mkdir(pathindir)
        os.mkdir(pathofblinkdata)
        maindf = pd.DataFrame(get_pageviews(target))
        maindf.to_csv(pathindir+ '/' + target + '.csv', index=False)
        for entry in new3:
            df = pd.DataFrame(get_pageviews(entry))
            if len(df) < (len(maindf)/2):
                pass
            else:
                df.to_csv(pathofblinkdata + '/' f'{entry}' + '.csv', index=False)


def filterbacklinks(listtofilter: list):
    if not isinstance(listtofilter, list):
        raise TypeError('Target has to be a list containing all Backlinks from one page')
    else:
        new = [k for k in listtofilter if not 'Benutzer' in k]
        new1 = [k for k in new if not 'Diskussion' in k]
        new2 = [k for k in new1 if not 'Wikipedia' in k]
        new3 = [k for k in new2 if not '/' in k]
        return new3



############################################################################################################
if __name__ == '__main__':

    views = []
    timestamp = []
    year = []

    # get data from one article
    data = get_pageviews("Meisen")

    # Get the timestamp of the first set of Data

    for entry in data:
        views.append((entry['views']))

    x = np.linspace(2015.5, 2022.5, num=len(views))
    plt.plot(x, views)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title("Meisen")
    plt.show()

    write_backlinks_tocsv("Wald-_und_Baumgrenze")

    '''
    #Getting all the Filtered Backlinks and then Writing them to an CSV for easier access
    for entry in new5:
        df = pd.DataFrame(get_pageviews(entry))
        if len(df) < 500:
            pass
        else:
            df.to_csv(path + '/' f'{entry}' + '.csv', index=False)
            print(df)

    
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
    sys.exit()

# Can remove unusefull links via:
# dataB.remove("Magdeburger Straßen/M")
# dataB.remove("Benutzer Diskussion:Nina/Archiv2")
# dataB.remove("Benutzer:Atamari/Liste der Vögel in Gambia")
