import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from ML_BASE.ML_operate.utils.tools import StandardScaler, load_data_timeindex, set_lag_missing, repeat_label_row
from ML_BASE.ML_operate.utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_CSCIDS2017(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        
        self.__read_data__()
    
    def __read_data__(self):
        self.scaler = MinMaxScaler()
        path = os.path.join(self.root_path, self.data_path)
        df = load_data_timeindex(path) # loader 변경
        cols = list(df.columns)
        
    
class Dataset_CSCIDS2017(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        pass