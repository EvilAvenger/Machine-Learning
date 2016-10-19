import sys
import numpy as np
import pandas as pd
from sklearn.svm import SVC
np.set_printoptions(suppress=True)



data = pd.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 3\\data\\svm-data.csv', names=None, header=None)

data = data.as_matrix().astype(float)
X = data[:,1:]
Y = data[0:, 0]


classifier = SVC(C=100000, random_state=241, kernel='linear',)
classifier.fit(X,Y)

print(data)
print(X)
print(Y)
print(classifier.support_)