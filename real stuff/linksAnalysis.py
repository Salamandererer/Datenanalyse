import pandas as pd
import requests
import matplotlib.pyplot as plt
from datetime import datetime
from urllib import parse


def get_links(article: str, prop: str):
    j = 0
    # props = links or linkshere"
    # links = forwardslinks
    # linkshere = backwardslinks

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


def to_links(data: dict, prop: str):
    data = data["query"]
    data = data["pages"]
    data = data[list(data.keys())[0]]
    return data[f"{prop}"]


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


####

data = get_links("Deutschland", "links")
print(data)

bundestagswahl = get_links("Bundestagswahl 2021", "links")
print(bundestagswahl)

bundestagswahl2 = get_forward_links("Bundestagswahl 2021")
print(bundestagswahl2)
print()

meisen = get_links("Meisen", "links")

df = pd.DataFrame(meisen)
df.to_csv('Meisen.csv', index=True, index_label="Index")
df.columns = ['referring links']
df['links to'] = 'Meisen'

print("MEISEN HIER BACKLINS", get_back_links("Meisen"))

print(df)

source = df['referring links'].tolist()

target = df['links to'].tolist()

links = list(zip(source, target))