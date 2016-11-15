#!/usr/bin/python3

import time
import sys
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from dataloader import DataLoader

__author__ = "Vitaly Bibikov"
__version__ = "0.1"

URL_DATA = 'https://github.com/EvilAvenger/Machine-Learning/blob/master/Yandex/Week%20Final/data/features.csv'
URL_TEST_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%20Final/data/features_test.csv'
FOLDER_NAME = 'data'
TARGET_COLUMN_NAME = 'radiant_win'

def __main__():
    data = get_data(URL_DATA, FOLDER_NAME)
    get_unique_heroes_count(data)

    # remove_cetegorial = False
    # replace_nans = lambda series: sys.maxsize # здесь нам удобнее отправить все ответы в какую-то отдельную ветку деревьев.
    # X, y = preprocess_data(data, replace_nans, remove_cetegorial)
    # learning_rates = [0.4]
    # n_estimators = [30]
    # run_grad_boost(X, y, n_estimators, learning_rates)

    data = get_data(URL_DATA, FOLDER_NAME)
    remove_cetegorial = True # убираем категориальные признаки для лин.регресии для лучшей работы
    replace_nans = lambda series: series.mean() #среднее по колонке работает лучше в случае лин.регрессии
    X, y = preprocess_data(data, replace_nans, remove_cetegorial)
    regularizators = [1] # субоптимальное значение  с масштабированием и удалением категориальных признаков
    scaling_enabled = True 
    run_log_regression(X, y, regularizators, scaling_enabled)

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

# Логистическая регрессия

def run_log_regression(X, y, regularizators, scaling_enabled):
    # без удаления категориальных и без масштабирования субоптимальны следующие значения:
    #regularizators = [0.4] 
    #scaling_enabled = False
    return regression_variations(X, y, regularizators, scaling_enabled)

def regression_variations(X, y, regularizators, scaling_enabled):
    """ Displays information about all iterations of the log regression with given C-regularizators """

    results = []
    for C in regularizators:
        values = log_regression(X, y, C, scaling_enabled)
        mean = np.mean(values)
        results.append(mean)

        print(values)
        print("Mean value: " + str(mean))
        print("Regularization step value: " + str(C))

    return results

@timer
@description_decorator("Cross validation for regression")
def log_regression(X, y, c, scaling_enabled):
    kfolds = KFold(len(y), n_folds=5, shuffle=True)

    params = []
    if scaling_enabled:
        params.append(StandardScaler())

    regressor = LogisticRegression(C=c, penalty='l2')
    params.append(regressor)
    clf = make_pipeline(*params)
    scores = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kfolds)
    return scores

# Градиентный бустинг

def run_grad_boost(X ,y, n_estimators, rate):
    """
    learning_rates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] (test)
    0.4 шаг обучения вышел наилучшим [0.1, 1] без применения масштабирования и предобработки

    n_estimators = [10, 20, 30, 35, 40, 45, 50] (test)
    Наблюдается незначительный рост с ростом количества решающих деревьев, но в ущерб времени исполнения.
    На мой взгляд проще оптимизировать другие параметры, чем нарашивать количество деревьев, если это возможно.
    """
    return grad_boost_variations(X, y, rate, n_estimators)


def grad_boost_variations(X, y, learning_rates, n_estimators):
    """ Displays information about all iterations of the gradient boosting with given parameters """
    results = []
    for rate in learning_rates:
        for estimator in n_estimators:
            values = grad_boost(X, y, rate, estimator)
            mean = np.mean(values)
            results.append(mean)
            print(values)
            print("Mean value: " + str(mean))
            print("Learning rate value: " + str(rate))
            print("Estimator value: " + str(estimator))

    return results


@timer
@description_decorator("Cross validation. Gradient Boosting")
def grad_boost(X, y, rate, num):
    """ Trains and performs cross validation of GradientBoosting Classifier """

    clf = GradientBoostingClassifier(n_estimators=num, verbose=True, learning_rate=rate)
    kfolds = KFold(len(y), n_folds=5, shuffle=True)
    scores = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kfolds)
    return scores

# Загрузка, обработка и удаление из данных того, что нам не нужно.

def preprocess_data(data, replace_nans, remove_cetegorial):
    remove_target_columns(data)
    process_nans(data, replace_nans)
    if remove_cetegorial:
        remove_categorial_data(data)
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

def remove_categorial_data(data):
    """ Removes columns which store categorial data """
    del data['lobby_type']
    for i in range(1, 6):
        del data['r{0}_hero'.format(i)]
        del data['d{0}_hero'.format(i)]

    return data

def get_unique_heroes_count(data):
    unique_set = set()
    for i in range(1, 6):
        dire = data['r{0}_hero'.format(i)].unique()
        radiant = data['d{0}_hero'.format(i)].unique()
        unique_set = unique_set | set(dire) | set(radiant)
    print("Unique hero count: {0}".format(len(unique_set)))
    return unique_set

# Обработка и вывод пропущенных значений (nan)

def process_nans(dataframe, replace_nans):
    columns, nans = get_nan_columns(dataframe)
    display_nan_values(columns, nans)
    replace_nan_values(dataframe, columns, replace_nans)

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


def replace_nan_values(dataframe, columns, replace_nans):
    """Replaces nans with maxsize (maxint) values """
    for name in columns:
        dataframe[name].fillna(replace_nans(dataframe[name]), inplace=True) # 0 or min value are suitable or dataframe[name].mean()
    return dataframe

def split_data(dataframe):
    """Splits data into  X and y """

    print("Target name: {0}".format(TARGET_COLUMN_NAME))
    y = dataframe[TARGET_COLUMN_NAME]
    del dataframe[TARGET_COLUMN_NAME]
    return (dataframe.values, y.values)





if __name__ == '__main__':
    __main__()
