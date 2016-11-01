from sklearn.decomposition import PCA
from dataloader import DataLoader

import numpy as np
import pandas as pd

URL_PRICES = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%204/data/close_prices.csv'
URL_INDEX = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%204/data/djia_index.csv'
FOLDER_NAME = 'data'

def __main__():
    
    data = get_data(URL_PRICES, FOLDER_NAME)
    (pca, result) = train_pca(get_matrix(data))

    num = get_first_indexies(pca.explained_variance_ratio_)
    print(num)
    
    dj_data  = get_matrix(get_data(URL_INDEX, FOLDER_NAME))
    correlation = count_pearson_corr(result[0:,0], dj_data)
    print(correlation)

    index = get_company_name(pca.components_)
    print(data.columns.values[index + 1])
    

def get_data(url, folder):
    loader = DataLoader()
    data = loader.download_data(url, folder)
    return data

def get_matrix(data):
     return data.as_matrix()[0:,1:]

def train_pca(data):
    pca = PCA(n_components=10)
    result = pca.fit_transform(data)
    return (pca, result)

def get_first_indexies(indexies):
    sum = i = 0
    while sum <= 0.90:
        sum = sum + indexies[i]
        i = i + 1
    return i

def get_company_name(values):
    values = values[0].tolist()
    return values.index(max(values))

def count_pearson_corr(data, djdata):
    djdata = list(np.reshape(djdata, len(djdata)))
    result = np.corrcoef(data, djdata)
    return round(result[0][1], 2)
    




























if __name__ == '__main__':
    __main__()
