#!/usr/bin/python3

import os
import copy
import time
import sys
import numpy as np
import pandas as pd
import urllib.request
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__author__ = "Vitaly Bibikov"
__version__ = "0.1"

URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%20Final/data/features.csv'
URL_TEST_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%20Final/data/features_test.csv'
FOLDER_NAME = 'data'
TARGET_COLUMN_NAME = 'radiant_win' # колонка с целевой переменной

def __main__():
    data = get_data(URL_DATA, FOLDER_NAME)
    test_data = get_data(URL_TEST_DATA, FOLDER_NAME)

    hero_number = get_unique_heroes_count(data)
    print("Unique hero count: {0}".format(hero_number))  #108

    run_grad_boost(data) # запускает все варианты градиентного бустинга из задания
    #run_log_regression(data) # запускает все варианты регрессии из задания

    #run_best_model(data, test_data)


#######################
# Декораторы
#######################

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

#######################
# Градиентный бустинг
#######################

def run_grad_boost(data):
    """ Функции с настройками для различных подпунктов финального задания
        в каждом методе заключены настройки для каждого этапа задания """
    #run_grad_boost_with_categorial(copy.deepcopy(data)) # Категориальные признаки остаются
    run_grad_boost_without_categorial(copy.deepcopy(data)) # Категориальные признаки удалены


@description_decorator("Cross validation for Grad boosting with categorial columns")
def run_grad_boost_with_categorial(data):

    """Execution time: 222.12287092208862
       [ 0.7012419   0.70165831  0.70445816  0.69854531  0.70398746]
       Mean value: 0.701978228739
       Learning rate value: 0.5
       Estimator value: 30 """

    create_bag = False
    remove_categorial = False
    replace_nans = lambda series: sys.maxsize # заменяем на большие значения
    rate = [0.5]
    n_estimators = [30]

    X, y = preprocess_data(data, replace_nans, remove_categorial, create_bag)  
    return grad_boost_variations(X, y, rate, n_estimators)

@description_decorator("Cross validation for Grad boosting without categorial data")
def run_grad_boost_without_categorial(data):

    """Execution time: 182.37156128883362
       [ 0.70214826  0.70136431  0.69899879  0.70098034  0.70459622]
       Mean value: 0.701617584784
       Learning rate value: 0.4
       Estimator value: 30  """

    create_bag = False
    remove_categorial = True
    rate = [0.4]
    n_estimators = [150]
    replace_nans = lambda series: sys.maxsize

    X, y = preprocess_data(data, replace_nans, remove_categorial, create_bag)  
    return grad_boost_variations(X, y, rate, n_estimators)

def grad_boost_variations(X, y, learning_rates, n_estimators):
    """ Displays information about all iterations 
    of the gradient boosting with given parameters """

    results = []
    for rate in learning_rates:
        for estimator in n_estimators:
            values = train_grad_boosting(X, y, rate, estimator)
            mean = np.mean(values)
            results.append(mean)
            print(values)
            print("Mean value: " + str(mean))
            print("Learning rate value: " + str(rate))
            print("Estimator value: " + str(estimator))

    return results

@timer
@description_decorator("Cross validation. Gradient Boosting")
def train_grad_boosting(X, y, rate, num):
    """ Trains and performs cross validation of GradientBoosting Classifier """
    clf = GradientBoostingClassifier(n_estimators=num, verbose=True, learning_rate=rate)
    kfolds = KFold(len(y), n_folds=5, shuffle=True)
    scores = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kfolds)
    return scores

#######################
# Логистическая регрессия
#######################

def run_log_regression(data):
    """ Функции с настройками для различных подпунктов финального задания
    в каждом методе заключены настройки для каждого этапа задания """
    log_regression_with_categorial(copy.deepcopy(data))  #категориальные признаки остаются
    log_regression_without_categorial(copy.deepcopy(data)) #категориальные признаки удалены
    log_regression_with_wordsbag(copy.deepcopy(data)) #категориальные признаки заменены на мешок (winner)

@description_decorator("Cross validation of log.regression with scaling only (1)")


def log_regression_with_categorial(data):
    """
    Execution time: 21.443242073059082
    [ 0.71585284  0.71348481  0.7134406   0.72158803  0.72101377]
    Mean value: 0.717076011537
    Regularization step value: 0.7"""
    
    scaling_enabled = True 
    remove_cetegorial = False 
    replace_nans = lambda series: series.mean() # среднее по колонке в данном случае работает лучше
    create_bag = False 
    regularizators = [0.7]

    X, y = preprocess_data(data, replace_nans, remove_cetegorial, create_bag)
    return regression_variations(X, y, regularizators, scaling_enabled)

@description_decorator("Cross validation of log.regression without categorial columns (2)")
def log_regression_without_categorial(data):
    """
    Execution time: 17.97003960609436
    [ 0.71683456  0.71607759  0.71987333  0.71779697  0.71478516]
    Mean value: 0.717073523269
    Regularization step value: 0.9"""

    scaling_enabled = True
    remove_cetegorial = True
    replace_nans = lambda series: series.mean()
    create_bag = False

    X, y = preprocess_data(data, replace_nans, remove_cetegorial, create_bag)
    regularizators = [0.9]
    return regression_variations(X, y, regularizators, scaling_enabled)

@description_decorator("Cross validation of log.regression with bag of words (4)")
def log_regression_with_wordsbag(data):
    """
    Execution time: 51.021955490112305
    [ 0.75098835  0.75783788  0.75236979  0.7546356   0.74509015]
    Mean value: 0.752184356492
    Regularization step value: 5
    """

    scaling_enabled = True
    remove_cetegorial = True
    replace_nans = lambda series: series.mean()
    create_bag = True

    X, y = preprocess_data(data, replace_nans, remove_cetegorial, create_bag)
    regularizators = [5]
    return regression_variations(X, y, regularizators, scaling_enabled)


def regression_variations(X, y, regularizators, scaling_enabled):
    """ Displays information about all iterations 
    of the log regression with given C-regularizators """

    for C in regularizators:
        values = train_log_regression(X, y, C, scaling_enabled)
        mean = np.mean(values)
        print(values)
        print("Mean value: " + str(mean))
        print("Regularization step value: " + str(C))

    return values

@timer
@description_decorator("Cross validation for regression")
def train_log_regression(X, y, c, scaling_enabled):
    clf = get_regression_classifier(c, scaling_enabled)
    kfolds = KFold(len(y), n_folds=5, shuffle=True)
    scores = cross_val_score(estimator=clf, X=X, y=y, scoring='roc_auc', cv=kfolds)
    return scores


def get_regression_classifier(C, scaling_enabled):
    classifiers = []
    if scaling_enabled:
        classifiers.append(StandardScaler())

    regressor = LogisticRegression(C=C, penalty='l2')
    classifiers.append(regressor)
    clf = make_pipeline(*classifiers)
    return clf

#######################
# Запуск наилучшей модели 
#######################

def run_best_model(data, test_data):
    """
    Логистическая регрессия с масштабированием признаков и использованием мешка слов. 
    Категориальные признаки удалены
    Пропущенные значения заменены на средние по столбцу
    Регулязиратор подобран с помощью GridSearchCV
    """

    scaling_enabled = True
    remove_cetegorial = True
    replace_nans = lambda series: series.mean()
    create_bag = True
    c_regularizator = 5

    X, y = preprocess_data(data, replace_nans, remove_cetegorial, create_bag)
    X_test = preprocess_data(test_data, replace_nans, remove_cetegorial, create_bag)[0]

    clf = get_regression_classifier(c_regularizator, scaling_enabled)
    clf.fit(X, y)
    result = clf.predict_proba(X_test)[:, 1]
    print_target(result)
    return result


def print_target(result):
    np.savetxt('result.csv', result, delimiter=',', fmt='%s')

#######################
# Загрузка, обработка и удаление из данных того, что нам не нужно.
#######################

def preprocess_data(data, replace_nans, remove_cetegorial, create_words_bag):
    """ Processes data with accordance to given parameters. """

    remove_target_columns(data)
    process_nans(data, replace_nans)
    words_bag_data = None

    if create_words_bag:
        words_bag_data = create_bag_of_words(data)

    if remove_cetegorial:
        data = remove_categorial_data(data)

    X, y = split_data(data)
    if words_bag_data is not None:
        X = np.column_stack((X, words_bag_data))

    return (X, y)

def get_data(url, folder):
    """Loads data to the specified folder from github source if they are not presented"""

    loader = DataLoader()
    data = loader.download_data(url, folder)
    return data

def remove_target_columns(data):
    """ Removes columns which are not required in the dataframe """

    delete_column(data, 'duration')
    delete_column(data, 'tower_status_radiant')
    delete_column(data, 'tower_status_dire')
    delete_column(data, 'barracks_status_radiant')
    delete_column(data, 'barracks_status_dire')
    return data

def delete_column(data, name):
    """ Deletes column from dataframe """

    is_deleted = False
    if name in data:
        del data[name]
        is_deleted = True

    return is_deleted

def remove_categorial_data(data):
    """ Removes columns which store categorial data """

    delete_column(data, 'lobby_type')
    for i in range(1, 6):
        delete_column(data, 'r{0}_hero'.format(i))
        delete_column(data, 'd{0}_hero'.format(i))

    return data

def get_unique_heroes_count(data):
    """ Counts number of unique heroes in dataset"""

    unique_set = set()
    for i in range(1, 6):
        dire = data['r{0}_hero'.format(i)].unique()
        radiant = data['d{0}_hero'.format(i)].unique()
        unique_set = unique_set | set(dire) | set(radiant)
    return len(unique_set)

def create_bag_of_words(data):
    """Creates bag of words out of categorial columns """

    hero_number = get_unique_heroes_count(data) + 4
    X_pick = np.zeros((data.shape[0], hero_number))
    for i, match_id in enumerate(data.index):
        for p in range(0, 5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1
    return X_pick

#######################
# Обработка и вывод пропущенных значений (nan)
#######################

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
        dataframe[name].fillna(replace_nans(dataframe[name]), inplace=True)
    return dataframe

def split_data(dataframe):
    """Splits data into  X and y """

    y = None
    print("Target name: {0}".format(TARGET_COLUMN_NAME))
    if TARGET_COLUMN_NAME in dataframe:
        y = dataframe[TARGET_COLUMN_NAME].values
        delete_column(dataframe, TARGET_COLUMN_NAME)

    return (dataframe.values, y)




class  DataLoader(object):

    def __init__(self):
        pass
        
    def _validate_data(self, url_name):
        result = os.path.exists(url_name)
        return result

    def _get_folder_path(self, url, folder_name):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file_name = url.rsplit('/')[-1]
        data_path = os.path.join(dir_path, folder_name, file_name)
        return data_path

    def download_data(self, url_name, folder_name):
        path = self._get_folder_path(url_name, folder_name)

        if not self._validate_data(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            urllib.request.urlretrieve(url_name, path)
        return pd.read_csv(path)


if __name__ == '__main__':
    __main__()
