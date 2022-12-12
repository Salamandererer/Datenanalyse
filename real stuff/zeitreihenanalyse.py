from datetime import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines as mlines

from sklearn.linear_model import LinearRegression

import linksAnalysis
from statsAnalysis import get_pageviews, analysis


def analyse(mainpage):
    mainviews = pageviewget(mainpage)
    backlinksMainview = linksAnalysis.get_back_links(mainpage)
    first20entrys = backlinksMainview[0:20]
    summe = []

    for entry in mainviews:
        summe.append(0)

    for entry in first20entrys:
        views = pageviewget(entry)
        differenceViews = [abs(m - km) for m, km in zip(mainviews, views)]
        newDifference = difference(mainviews, views, differenceViews)
        i = 1.00

        # diff = [abs(s - m) for s, m in zip(mainviews, views)]
        summe = [(km + bm) for km, bm in zip(summe, newDifference)]

        '''
        summe2 = summe

        while (sum(diff) / len(diff)) > 100:
            summe2 = [(i * km + i * bm) for km, bm in zip_longest(summe2, newDifference)]
            diff = [abs(s - m) for s, m in zip_longest(mainviews, summe2)]
            i = i - 0.01

        print("hahahahah")
        summe = summe2
        '''

    return summe


def pageviewget(name):
    syear = 2015
    start = datetime(syear, 1, 1)
    end = datetime.today()
    article = name

    data = get_pageviews(article, start, end, project="de.wikipedia.org")

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

    return views


def difference(views, views2, diffViews2):
    diffViews = (sum(diffViews2)) / len(views)

    for i in range(0, len(views2)):
        views2[i] = views2[i] + diffViews
    return views2


'''
summe = analyse("Meisen")
mainviews = pageviewget("Meisen")

x1 = np.linspace(2015, 2022.5, num=len(summe))
x2 = np.linspace(2015, 2022.5, num=len(mainviews))
plt.plot(x1, summe)
plt.plot(x2, mainviews)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("hilfe")
plt.show()
'''


def lineareRegression(article):
    mainviews = pageviewget(article)
    backlinksMainview = linksAnalysis.get_back_links(article)
    first20entrys = backlinksMainview[0:20]
    df = pd.DataFrame()
    len1 = len(mainviews)

    for entry in first20entrys:
        views = pageviewget(entry)
        print("länge", len(views))
        len2 = len(views)
        if len2 < len1:
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

    data = get_pageviews(article, datetime(2015, 1, 1), datetime.today(), project="de.wikipedia.org")
    data2 = analysis(data)

    x = df
    print(data2)
    y = mainviews

    model = LinearRegression()
    model.fit(x, y)

    x2 = np.linspace(2015, 2022.5, num=len(mainviews))
    '''
    intercept = model.intercept_[0]
    slope = model.coef_[0, 0]
    r_sq = model.score(x, y)
    print("intercept:", intercept)
    print("slope:", slope)
    print("coefficient of determination:", r_sq)
    '''

    yhat = model.predict(x)
    yhat[yhat < 0] = 0

    plt.scatter(mainviews, yhat)
    plt.title("Lineare Regression")
    plt.xlabel("Views Backlinks")
    plt.ylabel("Views: " + article)
    ax = plt.gca()
    line = mlines.Line2D([0, 1], [0, 1], color="red")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    plt.show()


lineareRegression("Meisen")

lineareRegression("Deutschland")

lineareRegression("Lineare Optimierung")

lineareRegression("Weihnachten")

lineareRegression("Köln")

lineareRegression("Großer Panda")


''' 
viewsMeisen = pageviewget("Meisen")
viewsKohlmeisen = pageviewget("Kohlmeise")
viewsBlaumeisen = pageviewget("Blaumeise")
viewsAmsel = pageviewget("Amsel")
viewsTannenmeisen = pageviewget("Tannenmeise")
viewsSchwanzmeisen = pageviewget("Schwanzmeise")

differenceKohlmeisen = [abs(m - km) for m, km in zip(viewsMeisen, viewsKohlmeisen)]
differenceBlaumeisen = [abs(m - km) for m, km in zip(viewsMeisen, viewsBlaumeisen)]
differenceAmsel = [abs(m - km) for m, km in zip(viewsMeisen, viewsAmsel)]
differenceTannenmeisen = [abs(m - km) for m, km in zip(viewsMeisen, viewsTannenmeisen)]
differenceSchwanzmeisen = [abs(m - km) for m, km in zip(viewsMeisen, viewsSchwanzmeisen)]

differenceKohlmeisen = difference(viewsMeisen, viewsKohlmeisen, differenceKohlmeisen)
differenceBlaumeisen = difference(viewsMeisen, viewsBlaumeisen, differenceBlaumeisen)
differenceAmsel = difference(viewsMeisen, viewsAmsel, differenceAmsel)
differenceTannenmeisen = difference(viewsMeisen, viewsTannenmeisen, differenceTannenmeisen)
differenceSchwanzmeisen = difference(viewsMeisen, viewsSchwanzmeisen, differenceSchwanzmeisen)


x = np.linspace(2015, 2022.5, num=len(viewsKohlmeisen))
plt.plot(x, viewsKohlmeisen)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("difference + alle views zusammen")
plt.show()
print("Max views", max(viewsMeisen))


summe = [(0.055*km + 0.055*bm) for km, bm in zip(differenceKohlmeisen, differenceBlaumeisen)]
summe = [(s + 0.055*a)for s, a in zip(summe, differenceAmsel)]
summe = [(s + 0.055*tm)for s, tm in zip(summe, differenceTannenmeisen)]
summe = [(s + 0.055*sm)for s, sm in zip(summe, differenceSchwanzmeisen)]

diff = [abs(s-m)for s, m in zip(summe, viewsMeisen)]

x = np.linspace(2015, 2022.5, num=len(summe))
plt.plot(x, summe)
plt.plot(x, viewsMeisen)
plt.plot(x, diff)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("difference + alle views zusammen")
plt.show()

summe2 = [(0.08*km + 0.08*bm) for km, bm in zip(differenceKohlmeisen, differenceBlaumeisen)]
summe2 = [(s + 0.08*a)for s, a in zip(summe2, differenceAmsel)]
summe2 = [(s + 0.08*tm)for s, tm in zip(summe2, differenceTannenmeisen)]
summe2 = [(s + 0.08*sm)for s, sm in zip(summe2, differenceSchwanzmeisen)]

diff2 = [abs(s-m)for s, m in zip(summe2, viewsMeisen)]

x = np.linspace(2015, 2022.5, num=len(summe2))
plt.plot(x, summe2)
plt.plot(x, viewsMeisen)
plt.plot(x, diff2)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("difference + alle views zusammen")
plt.show()

diff = sum(diff) / len(viewsMeisen)
medianMeise = sum(viewsMeisen) / len(viewsMeisen)
print(diff)
print(medianMeise)

'''