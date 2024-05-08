# from os import path
import pandas as pd
import json
import dataclasses


class pipe():
    def __init__(self):
        self.__factors = pd.read_excel('./DW/Storage/factors.xlsx', header=0).fillna('')

    def load_select_factor(self, service):
        return self.__factors # list(factors)
    
    def data_save(self, save_path, df):
        df.to_excel(save_path)

    def load_dw_data(self):
        pass
    
    def get_data_api(self):
        pass
    
    def info_save(self):
        pass
    
 
    
    
        