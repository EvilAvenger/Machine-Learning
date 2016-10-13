import io
import urllib.request

import numpy as np
import pandas as pd

from sklearn.cross_validation import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

link = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

biggest_num = 0
best_k = 0

def compare_biggest(i, num):
    global biggest_num 
    global best_k

    if biggest_num < num:
        biggest_num = num
        best_k = i

def print_formated(k, num):
    print("#{0}".format(k), format(num,'.2f'))
        
np.set_printoptions(suppress=True, precision=2)
data = pd.read_table(link, header=None, delimiter=',', dtype='float32')
values = data.as_matrix().astype(float)

X = values[0:,1:]
Y = values[:,0]

kfolds = KFold(len(Y), n_folds=5, shuffle=True, random_state=42)

print("W/O scaling: ", end='')

for i in range(1, 51):
    neighborsClassifier = KNeighborsClassifier(n_neighbors=i)
    clf = neighborsClassifier.fit(X, Y) 
    valScores = cross_val_score(estimator=clf, X=X, y=Y, scoring='accuracy', cv=kfolds)
    compare_biggest(i, valScores.mean())

print_formated(best_k, biggest_num)


print("With scaling: ", end='')

scaledX = scale(X.astype(float))
for i in range(1, 51):
    neighborsClassifier = KNeighborsClassifier(n_neighbors=i)
    clf = neighborsClassifier.fit(scaledX, Y) 
    valScores = cross_val_score(estimator=clf, X=scaledX, y=Y, scoring='accuracy', cv=kfolds)
    compare_biggest(i, valScores.mean())

print_formated(best_k, biggest_num)
