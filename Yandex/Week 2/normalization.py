import os
import sys

import numpy as np
import pandas

from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

test_data = pandas.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 2\\data\\perceptron-test.csv', header=None, dtype='float')
train_data = pandas.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 2\\data\\perceptron-train.csv', header=None, dtype='float')

np.set_printoptions(suppress=True)

test_data = test_data.as_matrix().astype(float)
train_data = train_data.as_matrix().astype(float)

X_train = train_data[0:, 1:]
Y_train = train_data[:, 0]


X_test = test_data[0:, 1:]
Y_test = test_data[:, 0]

clf = Perceptron(random_state=241)
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)

print(accuracy) #0.655

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

clf = Perceptron(random_state=241)
clf.fit(X_train_scaled, Y_train)

predictions = clf.predict(X_test_scaled)
accuracy_scaled = accuracy_score(Y_test, predictions)

print(accuracy_scaled) #0.845

delta_accuracy = accuracy_scaled - accuracy
print(delta_accuracy) #0.19