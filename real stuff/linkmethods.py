import glob
import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import os

from datetime import datetime
from urllib import parse
from bs4 import BeautifulSoup
from collections import Counter

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
    """
        gibt die Views als Zeitstrahl von einem eingegebenen Artikel zurück
    """
    try:
        data = get_pageviews(article)
        views = []
        for entry in data:
            views.append((entry['views']))
    except:
        print("No Data Availible for:", article)
    return views


def count_links(article: str):
    """
        Zählt die Anzahl der Links einer Seite mittels BeautifulSoup
    """

    soup = BeautifulSoup(requests.get(f"https://de.wikipedia.org/wiki/{article}").text, "html.parser")
    foundUrls = Counter(
        [link["href"] for link in soup.find_all("a", href=lambda href: href and not href.startswith("#"))])
    foundUrls = foundUrls.most_common()
    links = []

    for item in foundUrls:
        links.append("%s: %d" % (item[0], item[1]))
    return links


def get_target(target: str):
    """
        Gibt das Dataframe von einem eingegeben Artikel zurück
    """

    try:
        write_backlinks_tocsv(target)
    except:
        pass
    path = r'datafiles/' + target + '/' + target + '.csv'
    data = pd.read_csv(path)
    df = pd.DataFrame(data)[["timestamp", "views"]]
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    df = df.set_index("timestamp")
    return df


def get_backlink_views(target, ref_df):
    """
        Gibt die Views der Backlinks von einem eingegeben Target zurück
    """

    path = r'datafiles/' + target + '/backlinksdata'
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
    """

    """
    path = r'datafiles/' + article + '/backlinksdata'  # make path a global variable
    extension = 'csv'
    os.chdir(path)
    result = glob.glob('*.{}'.format(extension))
    cropped_result = [r[0:-4] for r in result]
    return cropped_result


def get_links(article: str, prop: str):
    """
    props:
    linkshere = Backlinks
    links = forwardlinks

    gibt die Forward oder Backwardlinks eines eingegebenen Artikels zurück
    """

    if prop == "links":
        limit = "pllimit=max"
        con = "plcontinue="
    else:
        limit = "lhlimit=max"
        con = "lhcontinue="

    titles = []
    data = requests.get(
        f"https://de.wikipedia.org/w/api.php?action=query&format=json&titles={article}&prop={prop}&{limit}").json()
    data_continue = data

    data = to_links(data, prop)
    for i in data:
        titles.append(i["title"])

    while list(data_continue.keys())[0] == "continue":
        data_continue = data_continue["continue"]
        data_continue = data_continue[con[0:len(con) - 1]]
        data_continue = requests.get(
            f"https://de.wikipedia.org/w/api.php?action=query&format=json&titles={article}&prop={prop}&{limit}&{con}{data_continue}").json()
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
    """
    Filtering the incoming list to filter out unwanted links to unnecessary Websites that have no Data or low amounts,
    then saving them to a relative path in the directory.
    """

    if not isinstance(target, str):
        raise TypeError('Target has to be a string')
    else:
        new0 = get_links(target, "linkshere")
        new = [k for k in new0 if not 'Benutzer' in k]
        new1 = [k for k in new if not 'Diskussion' in k]
        new2 = [k for k in new1 if not 'Wikipedia' in k]
        new3 = [k for k in new2 if not '/' in k]
        new4 = [k for k in new3 if not 'Kategorie' in k]
        new5 = [k for k in new4 if not target]
        pathindir = "datafiles/" + target
        pathofblinkdata = "datafiles/" + target + "/backlinksdata"
        try:
            os.mkdir(pathindir)
        except:
            print("Directory already exists")
            pass
        try:
            os.mkdir(pathofblinkdata)
        except:
            print("reading in the file")
            return
        maindf = pd.DataFrame(get_pageviews(target))
        maindf.to_csv(pathindir + '/' + target + '.csv', index=False)
        for entry in new5:
            df = pd.DataFrame(get_pageviews(entry))
            if len(df) < (len(maindf) / 2):
                pass
            else:
                df.to_csv(pathofblinkdata + '/' f'{entry}' + '.csv', index=False)


def filterlinks(listtofilter: list):
    """
        Filtert Links aus einer Liste von Backlinks heraus, die sonst die Predicition stören würden.
        Darunter zählen bspw. Benutzer.
    """
    if not isinstance(listtofilter, list):
        raise TypeError('Target has to be a list containing all Backlinks from one page')
    else:
        new = [k for k in listtofilter if not 'Benutzer' in k]
        new1 = [k for k in new if not 'Diskussion' in k]
        new2 = [k for k in new1 if not 'Wikipedia' in k]
        new3 = [k for k in new2 if not '/' in k]
        new4 = [k for k in new3 if not 'Kategorie' in k]
        return new4


def plottimeseries(article: str):
    """
        Plottet die Views der Zeitreihe für einen eingegeben Artikel.
    """
    # get data from one article
    data = get_pageviews(article)
    views = []
    for entry in data:
        views.append((entry['views']))
    # Plotting the Timeseries
    x = np.linspace(2015, 2022.5, num=len(views))
    plt.plot(x, views)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article)
    plt.show()


############################################################################################################
if __name__ == '__main__':
    plottimeseries("Meisen")
    plottimeseries("Kurvendiskussion")
    plottimeseries("Lyrik")
    plottimeseries("Programmiersprache")
    print(len(get_links("Programmiersprache", "linkshere")))
    print(len(get_links("Vögel", "linkshere")))
    print(len(get_links("Korrelation", "linkshere")))

    sys.exit()
