import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
import math

from dataloader import DataLoader
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss
from sklearn.ensemble import RandomForestClassifier

URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%205/data/gbm-data.csv'
FOLDER_NAME = 'data'


def __main__():
    learning_rates = [0.2]
    data = get_data()
    min = grad_boost(data, learning_rates)

    learning_rates = [1, 0.5, 0.3, 0.2, 0.1] 
    results = grad_boost(data, learning_rates)

    random_forest(data, min[0])
    return

def grad_boost(data, learning_rates):  
    X_train, X_test, Y_train, Y_test = split_data(data)
    results = []

    for rate in learning_rates:
        clf = train_model(X_train, Y_train, rate)
        train, test = get_decision_result(clf, X_train, X_test, Y_train, Y_test)
        results.append(analyze(train, test))

    return results

def random_forest(data, index):
    forest = RandomForestClassifier(n_estimators=index, random_state=241)
    X_train, X_test, Y_train, Y_test = split_data(data)
    forest.fit(X_train, Y_train)
    predictions = forest.predict_proba(X_test)
    result = log_loss(Y_test , predictions)
    return result


def get_data():
    loader = DataLoader()
    data = loader.download_data(URL_DATA, FOLDER_NAME)
    return data.values

def split_data(data):
    X = data[0:, 1:]
    Y = data[0:, 0]

    result = train_test_split(X, Y, test_size = 0.8, random_state=241)
    return result

def train_model(X, y, learning_rate):
    gbc = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=learning_rate)
    gbc.fit(X, y)
    return gbc

def get_decision_result(classifier, X_train, X_test, y_true_train, y_true_test):
    test_score = np.empty(len(classifier.estimators_))
    train_score = np.empty(len(classifier.estimators_))

    for i, pred in enumerate(classifier.staged_decision_function(X_train)):
        train_score[i] = count_loss(y_true_train, convert_descision(pred))

    for i, pred in enumerate(classifier.staged_decision_function(X_test)):
        test_score[i] = count_loss(y_true_test, convert_descision(pred))

    return (train_score, test_score)

def convert_descision(y):
    return  list(map(lambda x: 1 / (1 + math.exp(-float(x))), list(y)))


def count_loss(y_true, y_pred):
    return log_loss(y_true, y_pred)


def analyze(train, test):
    # plt.figure()
    # plt.plot(test, 'r', linewidth=2)
    # plt.plot(train, 'g', linewidth=2)
    # plt.legend(['test', 'train'])
    # plt.show()

    index = list(test).index(min(test)) + 1
    min_val = round(min(test), 2)

    print("{0} {1}".format(min_val, index))
    return index

if __name__ == '__main__':
    __main__()