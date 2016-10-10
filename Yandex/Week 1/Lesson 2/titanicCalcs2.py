import pandas
from sklearn.tree import DecisionTreeClassifier
import numpy as np


data = pandas.read_csv('D:\\Dropbox\\Study\\Programming\\Coursera\\Machine-Learning\\Yandex\\Week 1\\Lesson 1\\data\\titanic.csv', index_col='PassengerId')


del data['Name']
del data['SibSp']
del data['Parch']
del data['Ticket']
del data['Cabin']
del data['Embarked']

data['Sex'] = data['Sex'].replace('female', 0)
data['Sex'] = data['Sex'].replace('male', 1)

data = data.dropna(axis=0)



y = np.array(data['Survived'])
del data['Survived']

print(data)
X = np.array(data)

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
print(importances)

