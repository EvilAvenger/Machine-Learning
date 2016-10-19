import numpy as np
import pandas as pd
import sys

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.grid_search import GridSearchCV
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])


X = newsgroups.data
Y = newsgroups.target



tfid = TfidfVectorizer(input='content')
X_transformed = tfid.fit_transform(X)


# grid = {'C': np.power(10.0, np.arange(-5, 6))}

cv = KFold(Y.size, n_folds=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241, C=1.0)
clf.fit(X_transformed, Y)

# gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
# gs.fit(X_transformed, Y)

# for a in gs.grid_scores_:
#     print(a.mean_validation_score, end=': ')
#     print(a.parameters) # 0.993281075028: {'C': 1.0} - min C


indicies = np.argsort(np.abs(clf.coef_.toarray()))[0,-10:]

lst = list()
feature_mapping = tfid.get_feature_names()

for i in indicies:
    lst.append(feature_mapping[i])

for i in sorted(lst):
    print(i, end=' ') # atheism atheists bible god keith moon religion sci sky space 

    
