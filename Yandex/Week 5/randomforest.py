import numpy as np
import pandas as pd
import sys
from dataloader import DataLoader
 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


URL_DATA = 'https://raw.githubusercontent.com/EvilAvenger/Machine-Learning/master/Yandex/Week%205/data/abalone.csv'
FOLDER_NAME = 'data'


def __main__():
    loader = DataLoader()
    loader.download_data(URL_DATA, FOLDER_NAME)
    pass









if __name__ == '__main__':
    __main__()