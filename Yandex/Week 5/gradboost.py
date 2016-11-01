import numpy as np
import pandas as pd
import sys
from dataloader import DataLoader

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, cross_val_score

URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%205/data/gbm-data.csv'
FOLDER_NAME = 'data'


def __main__():
    data = get_data()
    print(data)


def get_data():
    loader = DataLoader()
    data = loader.download_data(URL_DATA, FOLDER_NAME)
    return data






if __name__ == '__main__':
    __main__()