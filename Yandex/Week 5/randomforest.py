import numpy as np
import pandas as pd
import sys
from dataloader import DataLoader

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score

URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%205/data/abalone.csv'
FOLDER_NAME = 'data'
SCORE_VALUE = 0.52

def __main__():
    X, y = get_data()

    results = []
    for i in range(1, 51):
        results.append(train_random_forest(X, y, i))
    
    for result in results:
        if result > SCORE_VALUE:
            print(result)
            print(results.index(result) + 1)
            break

    return

def get_data():
    loader = DataLoader()
    data = loader.download_data(URL_DATA, FOLDER_NAME)
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    data =  data.as_matrix()
    X = data[0:, 0:-1]
    y = data[0:, -1]
    return (X,y)


def train_random_forest(X, y, n_trees, ):
    
    folds = KFold(n=len(y), shuffle=True, random_state=1, n_folds=5)
    forest = RandomForestRegressor(random_state=1, n_estimators=n_trees)

    #clf = forest.fit(X, y) # version 1 of realization
    # scores = cross_val_score(estimator=clf, cv=folds, X=X, y=y)
   
    scores = []
    for train_index, test_index in folds:
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = y[train_index], y[test_index]
        forest.fit(X=X_train, y=Y_train)
        predictions = forest.predict(X_test)
        scores.append(r2_score(y_true = Y_test, y_pred = predictions))

    scores = np.mean(scores)
    
    return scores







if __name__ == '__main__':
    __main__()