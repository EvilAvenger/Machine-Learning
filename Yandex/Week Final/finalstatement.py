#!/usr/bin/python3
import time
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from dataloader import DataLoader

__author__ = "Vitaly Bibikov"
__version__ = "0.1"

URL_DATA = 'https://github.com/EvilAvenger/Machine-Learning/blob/master/Yandex/Week%20Final/data/features.csv'
URL_TEST_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%20Final/data/features_test.csv'
FOLDER_NAME = 'data'
TARGET_COLUMN_NAME = 'radiant_win'

def __main__():
    data = get_data(URL_DATA, FOLDER_NAME)
    run_grad_boost(data)
    run_log_regression(data)
    return

# Декораторы

def description_decorator(description):
    """Decorator that prints function description for display"""

    def wrapper(func):
        def decorated(*args, **kwargs):
            print("### {0} ###".format(description))
            return func(*args, **kwargs)
        return decorated
    return wrapper

def timer(func):
    """Decorator that prints execution time of a function"""

    def decorated(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        delta = time.time() - time_start
        print("Execution time: {0}".format(delta))
        return result
    return decorated

#Логистическая регрессия

def run_log_regression(data):
    pass

# Градиентный бустинг

def run_grad_boost(data):
    #learning_rates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
    # 0.4 шаг обучения вышел наилучшим на [0.1, 1]

    #n_estimators = [10, 20, 30, 35, 40, 45, 50] 
    #Есть смысл увеличивать N до 40 (хоть растет и совсем немного), дальше вообще небольшой рост. 
    #Возможно проще оптимизировать другие параметры, чем нарашивать количество деревьев.

    learning_rates = [0.4]
    n_estimators = [30]

    X, y = preprocess_data(data)
    grad_boost_variations(X, y, learning_rates, n_estimators)


def grad_boost_variations(X, y, learning_rates, n_estimators):
    """ Displays information about all iterations of the gradient boosting with given parameters """

    for rate in learning_rates:
        for estimator in n_estimators:
            result = grad_boost(X, y, rate, estimator)
            print(result)
            print("Mean: " + str(np.mean(result)))
            print("Learning rate: " + str(rate))
            print("estimator: " + str(estimator))
    return


@timer
@description_decorator("Cross validation")
def grad_boost(X, y, rate, num):
    """ Trains and performs cross validation of GradientBoosting Classifier """

    clf = GradientBoostingClassifier(n_estimators=num, verbose=True, learning_rate=rate)
    kfolds = KFold(len(y), n_folds=5, shuffle=True)
    scores = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kfolds)
    return scores

# Загрузка, обработка и удаление из данных того, что нам не нужно.

def preprocess_data(data):
    remove_target_columns(data)
    process_nans(data)
    X, y = split_data(data)
    return (X, y)

def get_data(url, folder):
    """Loads data to the specified folder from github source if they are not presented"""

    loader = DataLoader()
    data = loader.download_data(url, folder)
    return data

def remove_target_columns(data):
    """ Removes columns which are not required in the data """

    del data['duration']
    del data['tower_status_radiant']
    del data['tower_status_dire']
    del data['barracks_status_radiant']
    del data['barracks_status_dire']
    return data

# Обработка и вывод пропущенных значений (nan)

def process_nans(dataframe):
    columns, nans = get_nan_columns(dataframe)
    display_nan_values(columns, nans)
    replace_nan_values(dataframe, columns)

@description_decorator("Count of nan values")
def display_nan_values(columns, values):
    """ Prints columns collection to console and to csv file """

    for index, name in enumerate(columns):
        print('{0} - {1}'.format(name, values[index]))

    result = np.asarray([columns, values])
    np.savetxt('nan_statistics.csv', result, delimiter=',', fmt='%s')
    return

def get_nan_columns(dataframe):
    """Returns a tuple of columns with nan values and counts of nan per column"""

    names = dataframe.columns.tolist()
    nans = []

    for column in dataframe:
        nan_count = len(dataframe[column]) - dataframe[column].count()
        if nan_count > 0:
            nans.append(nan_count)
        else:
            names.remove(column)
    return (names, nans)

def replace_nan_values(dataframe, columns):
    """Replaces nans with maxsize (maxint) values """
    for name in columns:
        dataframe[name].fillna(sys.maxsize, inplace=True) # 0 or min value are suitable
    return dataframe

def split_data(dataframe):
    """Splits data into  X and y """

    print("Target name: {0}".format(TARGET_COLUMN_NAME))
    y = dataframe[TARGET_COLUMN_NAME]
    del dataframe[TARGET_COLUMN_NAME]
    return (dataframe.values, y.values)





if __name__ == '__main__':
    __main__()
