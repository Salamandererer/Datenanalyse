from urllib import parse

import numpy as np
import requests
from datetime import datetime

i = 0
c = 2663
c2 = 2377
b = 365
o = [1, 2, 3, 4, 5]
p = np.array([1, 2, 3, 4, 5])

e = (b % c)
print(c // b)
e = (divmod(c, b))
while c2 > 365:
    c2 = c2 - 365
    i = i + 1
print(i)
print(e[0])
help(requests.Session)


def get_links(article: str, prop: str):
    "props = links or linkshere"

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


print(get_links("Meisen", "links"))
print("lol")


def get_better_links(article: str):
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


print(get_better_links("Meisen"))

URL = "https://wikimedia.org/api/rest_v1/"
HEADERS = {"Accept": "application/json", "user-agent": "marvin.braun1@smail.inf.h-brs.de"}


def get_pageview2s(article: str, start: datetime, end: datetime, project="de.wikipedia.org",
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

print("Get Pageviews = ")
start2 = datetime(2000, 1, 1)
end2 = datetime.today()
print(type(get_pageview2s("Meisen", start2, end2))) #list