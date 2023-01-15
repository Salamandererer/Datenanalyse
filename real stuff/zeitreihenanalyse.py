import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from pandas.core import frame
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from linkmethods import get_views,get_links


def lineareRegression(article):
    mainviews = get_views(article)
    backlinksMainview = get_links(article, "linkshere")
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    len1 = len(mainviews)

    for entry in backlinksMainview:  # for every backlink from the mainpage
        views = get_views(entry)  # get the views
        len2 = len(views)
        if len2 < len1:  # then compare if they have the same amount of entries
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

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
    differenceViews = smape(mainviews, yhat)
    plt.plot(x1, differenceViews)
    plt.xlabel("Difference " + article + " to all Backlinks")
    plt.ylabel("Average Error")
    plt.show()

    plt.plot(x1, yhat)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article + " Lineare Regression")
    plt.show()

    plt.boxplot(differenceViews)
    plt.title("Lin Regression")
    plt.ylabel("Average Error")
    plt.xlabel(article)
    plt.show()

    print("SMAPE Lin Regression: ", differenceViews.mean())

    return intercept, slope, r_sq


def logisticRegression(article):
    mainviews = get_views(article)
    backlinksMainview = get_links(article, "linkshere")
    df = pd.DataFrame()
    len1 = len(mainviews)
    for entry in backlinksMainview:
        views = get_views(entry)
        len2 = len(views)
        if len2 < len1:
            diff = len1 - len2
            for i in range(0, diff):
                views.append(0)

        df.insert(loc=0, column=entry, value=views)

    x = df
    y = mainviews

    model = LogisticRegression()
    model.fit(x, y, 0.2)

    x2 = np.linspace(2015, 2022.5, num=len(mainviews))

    intercept = model.intercept_
    slope = model.coef_
    r_sq = model.score(x, y)

    print("intercept:", intercept)
    print("slope:", slope)
    print("coefficient of determination:", r_sq)

    yhat = model.predict(x)
    yhat[yhat < 0] = 0

    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=0.3,
                                                        random_state=None)

    x1 = np.linspace(2015, 2022.5, num=len(yhat))

    plt.plot(x1, yhat)
    plt.xlabel("Time")
    plt.ylabel("Views")
    plt.title(article + " Logarithmisch Lineare Regression")
    plt.show()

    # relativer fehler
    differenceViews = smape(mainviews, yhat)
    plt.plot(x1, differenceViews)
    plt.xlabel("Difference " + article + " to all Backlinks")
    plt.ylabel("Average Error")
    plt.show()

    plt.boxplot(smape(mainviews, yhat))
    plt.title("Log Regression")
    plt.title("Log Regression")
    plt.ylabel("Average Error")
    plt.xlabel(article)
    plt.show()

    print("SMAPE Log-Lin Regression: ", differenceViews.mean())

def smape(target, forecast):
    if type(target) == pd.core.frame.DataFrame:
        target = target.values

    denominator = np.abs(target) + np.abs(forecast)
    flag = denominator == 0.

    smape = 2 * (
            (np.abs(target - forecast) * (1 - flag)) / (denominator + flag)
    )
    return smape


if __name__ == '__main__':
    article = "Bundesrat_(Deutschland)"
    lineareRegression(article)
    logisticRegression(article)
