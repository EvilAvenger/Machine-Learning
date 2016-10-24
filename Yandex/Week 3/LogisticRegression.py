
#!/usr/bin/python3

import math
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


np.set_printoptions(suppress=True)
data = pd.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 3\\data\\data-logistic.csv', names=None, header=None)

data = data.as_matrix().astype(float)

X = data[:, 1:]
y = data[:, 0]


lr = LogisticRegression(C=10, fit_intercept=True, intercept_scaling=0.1, penalty='l2')
clf_l2_LR = lr.fit(X, y)
coef_l2_LR = clf_l2_LR.score(X,y)

print("{0:.3f}".format(coef_l2_LR))



#no - 0.727
#no - 0.873