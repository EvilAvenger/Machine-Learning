import sys
import urllib.request
import os
import pandas as pd

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