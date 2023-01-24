import pandas as pd
import numpy as np
import sklearn.preprocessing

from matplotlib import pyplot as plt
from matplotlib import lines as mlines
from pandas.core import frame
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from linkmethods import get_views, get_links, filterlinks

cutoff = -512


def lineareRegression(article):
    mainviews = get_views(article)[cutoff:-1]
    backlinksMainview = filterlinks(get_links(article, "linkshere"))
    df = pd.DataFrame()
    df2 = pd.DataFrame()
    len1 = len(mainviews)

    for entry in backlinksMainview:  # for every backlink from the mainpage
        views = get_views(entry)[cutoff:-1]  # get the views but cutting the list at a certain point
        # to compare it with the NN
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
                                                        test_size=0.15,
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

    x1 = np.linspace(2021.1, 2022.5, num=len(yhat))

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
    # in this file we added the cutoff for the lists to have the same data we use in the Neural network
    article = "Depression"
    print("Working on...", article)
    lineareRegression(article)
