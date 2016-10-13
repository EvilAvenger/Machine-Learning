import numpy as np
import pandas as pd
import sklearn.datasets as skd

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale


biggest_num = -100
best_k = -100

def compare_biggest(i, num):
    global biggest_num 
    global best_k

    if biggest_num < num:
        biggest_num = num
        best_k = i

def print_formated(k, num):
    print("#{0}".format(k), format(num,'.2f'))


dataset = skd.load_boston() 

scaledX = scale(dataset.data)
Y = dataset.target

kfolds = KFold(len(Y), n_folds=5, shuffle=True, random_state=42)
samples = np.linspace(1, 10, num=200)

for i in samples:
    clf = KNeighborsRegressor(n_neighbors=5, p=i, metric='minkowski', weights='distance')
    estimator = clf.fit(scaledX, Y)
    result = cross_val_score(estimator=estimator, X=scaledX, y=Y, scoring='mean_squared_error', cv=kfolds)
    print(i, result.mean())
    compare_biggest(i, result.mean())

print('Best:', end='')
print_formated(best_k, biggest_num)

#9.50251256281407 -21.11