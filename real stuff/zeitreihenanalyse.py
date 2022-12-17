from datetime import datetime
from itertools import zip_longest
from statistics import mean
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from pandas.core import frame
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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
        len2 = len(views)
        if len2 < len1:
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

    data = get_pageviews(article, datetime(2015, 1, 1), datetime.today(), project="de.wikipedia.org")
    data2 = analysis(data)

    x = df
    y = mainviews

    model = LinearRegression()
    model.fit(x, y, 0.2)

    x2 = np.linspace(2015, 2022.5, num=len(mainviews))

    intercept = model.intercept_
    slope = model.coef_[0]
    r_sq = model.score(x, y)

    print("intercept:", intercept)
    print("slope:", slope)
    print("coefficient of determination:", r_sq)

    yhat = model.predict(x)
    yhat[yhat < 0] = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=None)
    lr = LinearRegression()
    lr.fit(x_train, y_train)

    print('------ Lineare Regression -----')
    print('Funktion via sklearn: y = %.3f * x + %.3f' % (lr.coef_[0], lr.intercept_))
    print("Alpha: {}".format(lr.intercept_))
    print("Beta: {}".format(lr.coef_[0]))
    print("Training Set RÂ² Score: {:.2f}".format(lr.score(x_train, y_train)))
    print("Test Set RÂ² Score: {:.2f}".format(lr.score(x_test, y_test)))
    print("\n")

    plt.scatter(mainviews, yhat, alpha=0.7)
    plt.title("Lineare Regression")
    plt.xlabel("Views Backlinks")
    plt.ylabel("Views: " + article)
    ax = plt.gca()
    line = mlines.Line2D([0, 1], [0, 1], color="red")
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    print(ax)
    plt.show()

    x1 = np.linspace(2015, 2022.5, num=len(yhat))
    plt.plot(x1, yhat)
    print("AAAAAAAAAAAAAAAAAAAAAA", len(yhat))
    print("BBBBBBBBBBBBBBBBB", len(mainviews))
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title("Meisen Regression YHAT")
    plt.show()

    return intercept, slope, r_sq


lineareRegression("Meisen")


def multLinRegression(article):
    mainviews = pageviewget(article)
    backlinksMainview = linksAnalysis.get_back_links(article)
    first20entrys = backlinksMainview[0:20]
    df = pd.DataFrame()
    len1 = len(mainviews)

    for entry in first20entrys:
        views = pageviewget(entry)
        len2 = len(views)
        if len2 < len1:
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

    data = get_pageviews(article, datetime(2015, 1, 1), datetime.today(), project="de.wikipedia.org")
    data2 = analysis(data)

    x = df
    y = mainviews

    model = LinearRegression()
    model.fit(x, y)

    intercept = model.intercept_
    slope = model.coef_

    print(len(slope))
    return intercept, slope


def berechneDaten(alpha, beta, entrys):
    summe = []
    for entry in entrys[0:20]:
        for views in range(0, len(entry)):
            summe.append((alpha * views + beta))

    return summe


def berechneSumme(alpha, beta, summe, scr, abw):
    i = 0
    print(max(summe))

    for entry in summe[0: len(summe)]:
        if entry > 500 * abw:
            entry = mean(summe)
            print(entry)

        summe[i] = (alpha + beta * entry) * (1 - scr)

        i += 1
    return summe


def berechneSumme2(alpha, beta, backlinks):
    i = 0
    print(max(backlinks))
    print(len(backlinks))
    initViews = pageviewget(backlinks[0])

    summe = []
    for x in range(0, len(initViews)):
        summe.append(0)

    entrys = []
    for y in range(0, len(initViews)):
        entrys.append(0)

    for entry in backlinks[0: len(backlinks)]:
        views = pageviewget(entry)

        for view in views[0: len(views)]:
            print("BETAAAAA", beta[i])
            print("ENTRYYYSSSS", len(entrys))
            print("VIIIEEWWWWSSSS", len(views))

            if len(entrys) > len(views):
                diff = len(entrys) - len(views)
                for i in range(0, diff):
                    views.append(0)

            print("KEYYYYYYYYYYYY", view)
            entrys = [(e + beta[i] * v) for e, v in zip_longest(entrys, views)]

        summe = [(s + e + alpha) for s, e in zip_longest(summe, entrys)]
        i += 1
    return summe


def abweichungsfaktor(summe, mainview):
    abw = [abs(s1 - m) for s1, m in zip(summe, mainview)]
    abw1 = (sum(abw)) / len(abw)
    return abw1


alphaMeisen, betaMeisen, score = lineareRegression("Meisen")
alpha2, beta2, score2 = lineareRegression("Schmetterlinge")

a, b = multLinRegression("Meisen")


def smape(target, forecast):
    if type(target) == pd.core.frame.DataFrame:
        target = target.values

    denominator = np.abs(target) + np.abs(forecast)
    flag = denominator == 0.

    smape = 2 * (
            (np.abs(target - forecast) * (1 - flag)) / (denominator + flag)
    )
    return smape


def MAPE(target, predicted):
    def mape(actual, pred):
        actual, pred = np.array(actual), np.array(pred)
        return (np.abs((actual - pred) / actual)) * 100


'''
alpha, beta = lineareRegression("Deutschland")

alpha, beta = lineareRegression("Lineare Optimierung")

alpha, beta = lineareRegression("Weihnachten")

alpha, beta = lineareRegression("Köln")

alpha, beta = lineareRegression("Großer Panda")


mainviews = pageviewget("Meisen")
backlinksMainview = linksAnalysis.get_back_links("Meisen")
first20entrys = backlinksMainview[0:20]
summe1 = analyse("Meisen")
abw2 = abweichungsfaktor(summe1, mainviews)
su = berechneSumme(alphaMeisen, betaMeisen, summe1, score, abw2)

x1 = np.linspace(2015, 2022.5, num=len(su))
plt.plot(x1, su)
print("AAAAAAAAAAAAAAAAAAAAAA", len(su))
print("BBBBBBBBBBBBBBBBB", len(mainviews))
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Meisen Regression")
plt.show()

x2 = np.linspace(2015, 2022.5, num=len(mainviews))
plt.plot(x2, mainviews)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Meisen normal")
plt.show()

##############

mainviews4 = pageviewget("Meisen")
backlinksMainview4 = linksAnalysis.get_back_links("Meisen")
first20entrys4 = backlinksMainview4[0:20]
summe4 = analyse("Meisen")
abw4 = abweichungsfaktor(summe4, mainviews)
su4 = berechneSumme2(a, b, first20entrys4)

x1 = np.linspace(2015, 2022.5, num=len(su4))
plt.plot(x1, su4)
plt.xlabel("Time")
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Meisen Regression HILFE")
plt.show()

x2 = np.linspace(2015, 2022.5, num=len(mainviews))
plt.plot(x2, mainviews4)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Meisen normal HILFE")
plt.show()

#######

mainviews2 = pageviewget("Schmetterlinge")
backlinksMainview2 = linksAnalysis.get_back_links("Schmetterlinge")
first20entrys2 = backlinksMainview2[0:20]
summe2 = analyse("Schmetterlinge")
abw2 = abweichungsfaktor(summe1, mainviews)
su2 = berechneSumme(alpha2, beta2, summe2, score2, abw2)

x3 = np.linspace(2015, 2022.5, num=len(su2))
plt.plot(x3, su2)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Schmetterlinge Regression")
plt.show()

x4 = np.linspace(2015, 2022.5, num=len(mainviews2))
plt.plot(x4, mainviews2)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Schmetterlinge normal")
plt.show()

su.append(0)
su = pd.DataFrame(su, columns=['predicted'])
mw = pd.DataFrame(mainviews, columns=['views'])
print("type su", su)
print("type mv", mw)
sm = smape(mw, su)
print("mape ier", MAPE(mw['views'], su['predicted']))
sm['mape'] = MAPE(mw['views'], su['predicted'])
print("smape hier lol", sm.mean())

x1 = np.linspace(2015, 2022.5, num=len(s))
x2 = np.linspace(2015, 2022.5, num=len(mainviews))
plt.plot(x1, s)
plt.plot(x2, mainviews)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Meisen Regression")
plt.show()

mainviews2 = pageviewget("Deutschland")
backlinksMainview2 = linksAnalysis.get_back_links("Deutschland")
first20entrys2 = backlinksMainview2[0:20]
s2 = berechneDaten(alphaDeutschland, betaDeutschland, first20entrys2)

x1 = np.linspace(2015, 2022.5, num=len(s2))
x2 = np.linspace(2015, 2022.5, num=len(mainviews2))
plt.plot(x1, s2)
plt.plot(x2, mainviews2)
plt.xlabel("Time")
plt.ylabel("Views")
plt.title("Deutschland Regression")
plt.show()

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
