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
from linkmethods import get_pageviews, get_back_links
from testplots import analysis


def analyse(mainpage):
    mainviews = pageviewget(mainpage)
    backlinksMainview = get_back_links(mainpage)
    summe = []

    for entry in mainviews:
        summe.append(0)

    for entry in backlinksMainview:
        views = pageviewget(entry)
        differenceViews = [abs(m - km) for m, km in zip(mainviews, views)]
        newDifference = difference(mainviews, views, differenceViews)
        i = 1.00

        diff = [abs(s - m) for s, m in zip(mainviews, views)]
        summe = [(km + bm) for km, bm in zip(summe, newDifference)]

    return summe


def pageviewget(article):
    views = []
    data = []

    try:
        data = get_pageviews(article, project="de.wikipedia.org")
    except:
        pass

    for entry in data:
        views.append((entry['views']))

    return views


def difference(views, views2, diffViews2):
    diffViews = (sum(diffViews2)) / len(views)

    for i in range(0, len(views2)):
        views2[i] = views2[i] + diffViews
    return views2



def lineareRegression(article):
    mainviews = pageviewget(article)
    backlinksMainview = get_back_links(article)
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    len1 = len(mainviews)

    for entry in backlinksMainview:  # for every backlink from the mainpage
        views = pageviewget(entry)  # get the views
        len2 = len(views)
        if len2 < len1:  # then compare if they have the same amount of entries
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

    data = get_pageviews(article, project="de.wikipedia.org")
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
    plt.show()

    x1 = np.linspace(2015, 2022.5, num=len(yhat))

    # relativer fehler
    differenceViews = [abs(m - km) / m for m, km in zip(mainviews, yhat)]
    plt.plot(x1, differenceViews)
    plt.xlabel("Difference Meisen to all Backlinks")
    plt.ylabel("Average Error")
    print("Kommutativen relativ fehler", np.mean(differenceViews))
    plt.show()

    plt.plot(x1, yhat)
    print("AAAAAAAAAAAAAAAAAAAAAA", len(yhat))
    print("BBBBBBBBBBBBBBBBB", len(mainviews))
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title("Meisen Regression YHAT")
    plt.show()

    plt.boxplot(differenceViews)
    plt.ylabel("Average Error")
    plt.xlabel("Meisen")
    plt.show()

    return intercept, slope, r_sq


lineareRegression("Meisen")

if __name__ == '__main__':
    lineareRegression("Meisen")


def multLinRegression(article):
    mainviews = pageviewget(article)
    backlinksMainview = get_back_links(article)
    first20entrys = backlinksMainview
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

    data = get_pageviews(article, project="de.wikipedia.org")
    data2 = analysis(data)

    x = df
    y = mainviews

    model = LinearRegression()
    model.fit(x, y)

    intercept = model.intercept_
    slope = model.coef_

    print("lenSlope", len(slope))
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
        if entry > 250 * abw:
            entry = mean(summe)
            print(entry)

        summe[i] = (alpha + beta * entry) * (1 - scr)

        i += 1
    return summe


def berechneSumme2(alpha, beta, backlinks):
    i = 0
    print("Maxbacklinks", max(backlinks))
    print("lenofbacklinks", len(backlinks))
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

