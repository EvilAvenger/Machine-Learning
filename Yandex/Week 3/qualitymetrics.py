from itertools import groupby
from collections import Counter

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

import numpy as np
import pandas as pd



def __main__():
    np.set_printoptions(suppress=True)
    classes = get_classification_classes()
    metrics = get_metrics()
    scores = get_scores()
    results = get_precision_recall_max()

def get_classification_data():
    data = pd.read_csv('https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%203/data/classification.csv', names=None, header=None).as_matrix()
    return data[1:]

def get_scored_data():
    data = pd.read_csv('https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%203/data/scores.csv', names=None, header=None).as_matrix()
    return data[1:]


def get_classification_classes():
    data = get_classification_data()
    classes = []  
    for row in data:
        record_class = get_class(row)       
        classes.append(record_class)

    counted_list = Counter(classes)

    result = '{TP} {FP} {FN} {TN}'.format(TP=counted_list['TP'], FP=counted_list['FP'], FN=counted_list['FN'], TN=counted_list['TN'])
    print(result)

    return counted_list

def get_metrics():
    data = get_classification_data()
    y_true = list(map(int, data[0:, 0]))
    y_pred = list(map(int, data[0:, 1]))

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print_formated(accuracy, precision, recall, f1)
    print()
    return (accuracy, precision, recall, f1)

def get_scores():

    data = get_scored_data()

    y_true = list(map(int, data[0:, 0]))
    score_logreg = list(map(float, data[0:, 1]))
    score_svm = list(map(float, data[0:, 2]))
    score_knn = list(map(float, data[0:, 3]))
    score_tree = list(map(float, data[0:, 4]))

    score_logreg = roc_auc_score(y_true, score_logreg)
    score_svm = roc_auc_score(y_true, score_svm)
    score_knn = roc_auc_score(y_true, score_knn)
    score_tree = roc_auc_score(y_true, score_tree)
    
    return print_max(score_logreg, score_svm, score_knn, score_tree) 

def get_precision_recall_max():

    data = get_scored_data()

    y_true = list(map(int, data[0:, 0]))
    score_logreg = list(map(float, data[0:, 1]))
    score_svm = list(map(float, data[0:, 2]))
    score_knn = list(map(float, data[0:, 3]))
    score_tree = list(map(float, data[0:, 4]))

    score_logreg = get_concrete_precision(y_true, score_logreg)
    score_svm = get_concrete_precision(y_true, score_svm)
    score_knn = get_concrete_precision(y_true, score_knn)
    score_tree = get_concrete_precision(y_true, score_tree)
    results = print_max(score_logreg, score_svm, score_knn, score_tree)

    return results

def get_concrete_precision(y_true,score):
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    filtered_recall = [num for num in recall if num >= 0.7]
    filtered_precision = precision[:len(filtered_recall)]
    return max(filtered_precision)

def get_class(data):
    def swicth(data):
        return {
            '00': 'TN',
            '01': 'FP',
            '10': 'FN',
            '11' :'TP'
        }.get(data, 'None')
    data = data[0] + data[1]
    data = swicth(data)
    return data

def print_max(score_logreg, score_svm, score_knn, score_tree):
    results = {'score_logreg' : score_logreg, 'score_svm': score_svm, 'score_knn' : score_knn, 'score_tree':score_tree}
    print(max(results, key=results.get))
    return results

def print_formated(*args):
    for arg in args:
        print(format(arg, '.2f'), end=' ')










if __name__ == '__main__':
    __main__()