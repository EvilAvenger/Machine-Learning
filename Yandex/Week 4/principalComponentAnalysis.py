import urllib.request
import os
import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%204/data/salary-train.csv'
URL_TEST_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%204/data/salary-test-mini.csv'
FOLDER_DATA = 'data'


def __main__():
    loader = DataLoader()

    data = loader.download_data(URL_DATA, FOLDER_DATA)
    data = preprocess_text(data)
    
    tfid = TfidfVectorizer(input='content', min_df=5)
    vectorized_data = tfid.fit_transform(data['FullDescription'])

    vectorizer = DictVectorizer()
    dict_vectorized = vectorizer.fit_transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))

    Y = get_target_data(data)
    X = hstack([vectorized_data, dict_vectorized])

    ridge = Ridge(alpha=1.0, random_state=241)
    ridge.fit(X, Y)

    data = loader.download_data(URL_TEST_DATA, FOLDER_DATA)
    data = preprocess_text(loader.download_data(URL_TEST_DATA, FOLDER_DATA)) 
    vectorized_data = tfid.transform(data['FullDescription'])
    dict_vectorized = vectorizer.transform(data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_train = hstack([vectorized_data, dict_vectorized])

    result = ridge.predict(X_train)

    print(result)

    

def preprocess_text(data):
    regex = re.compile('[^a-zA-Z0-9]')
    data['FullDescription'] = data['FullDescription'].replace(regex, ' ', regex=True).str.lower()
    data['LocationNormalized'].fillna('nan', inplace=True)
    data['ContractTime'].fillna('nan', inplace=True)
    return data



def get_target_data(data):
    return data['SalaryNormalized']


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
            urllib.request.urlretrieve(url_name, path)
        return pd.read_csv(path)







if __name__ == '__main__':
    __main__()