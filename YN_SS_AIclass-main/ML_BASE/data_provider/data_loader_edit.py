import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
# from utils.tools import StandardScaler, load_data_timeindex, set_lag_missing, repeat_label_row
# from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')


class Dataset_BIVA(Dataset):
    def __init__(self, root_path, flag='train', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
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
        self.period = {'M': ['2000-01','2023-06'], 'Q':['2000-01','2023-06']}
        self.start_M = self.period['M'][0]
        self.end_M = self.period['M'][1]
        self.start_Q = self.period['Q'][0]
        self.end_Q = self.period['Q'][1]
        
#        self.load_data_timeindex = load_data_timeindex
#        self.set_lag_missing = set_lag_missing
#        self.repeat_label_row = repeat_label_row
        
        self.__read_data__()


    def __read_data__(self):
        self.scaler_m = MinMaxScaler()
        # self.scaler_q = MinMaxScaler()
        
        path = os.path.join(self.root_path, self.data_path)
        df_Q, _, df_M, _, self.var_info = self.load_data_timeindex(path)
        
        cols_Q = list(df_Q.columns)
        cols_Q.remove(self.target)
        df_Q = df_Q[cols_Q + [self.target]]
        df_M = df_M.loc[self.start_M:self.end_M]
        df_Q = df_Q.loc[self.start_Q:self.end_Q]
        # temp Q_variable insert
        df_M = pd.concat([df_M,df_Q],axis=1)
        cols_M = list(df_M.columns)

        df_Q = self.repeat_label_row(df=df_Q,pred_len=self.pred_len,repeat=3)
        
        num_train = int(len(df_M) * 0.80) 
        num_test = int(len(df_M) * 0.10)
        num_vali = len(df_M) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_M) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_M) - (self.pred_len - 1)*3] # quart feq(3) * pred period 
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        border1_v = border1 * self.pred_len
        border2_v = border2 * self.pred_len
        # print(f"border1,border2:{border1},{border2}")
        # print(f"border1_v,border2_v:{border1_v},{border2_v}")
        
        if self.set_type == 0:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
                
        elif self.set_type == 1:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
        else:
            if self.features == 'M' or self.features == 'MS':
                # cols_data = df_M.columns[1:]
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]
            elif self.features == 'S':
                df_data = df_M[cols_M]
                df_data_t = df_Q[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            df_data_cols = df_data.columns
            df_data_index = df_data.index
            
            train_data_t = df_data_t[border1s[0]*self.pred_len:border2s[0]*self.pred_len]
            df_data_t_cols = df_data_t.columns
            df_data_t_index = df_data_t.index
            
            self.scaler_m.fit(train_data.values)
            data = self.scaler_m.fit_transform(df_data.values)
            # self.scaler_q.fit(train_data_t.values)
            # data_t = self.scaler_q.fit_transform(df_data_t.values)
            
            data = pd.DataFrame(data,columns=df_data_cols, index=df_data_index)
            data_t = df_data_t # pd.DataFrame(data_t,columns=df_data_t_cols, index=df_data_t_index)
            
        else:
            data = df_data.values
            data_t = df_data_t.values

        # data_stamp_data_M
        df_stamp =  data.index
        # df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
#            data_stamp = time_features(pd.to_datetime(df_stamp.values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
            data_stamp = df_stamp.values

        # data_stamp_target_Q
        df_stamp_t = data_t.index # df_Q.index[border1:border2]
        # df_stamp_t['date'] = pd.to_datetime(df_stamp_t.date)
        if self.timeenc == 0:
            df_stamp_t['month'] = df_stamp_t.apply(lambda row: row.month, 1)
            df_stamp_t['day'] = df_stamp_t.apply(lambda row: row.day, 1)
            df_stamp_t['weekday'] = df_stamp_t.apply(lambda row: row.weekday(), 1)
            df_stamp_t['hour'] = df_stamp_t.apply(lambda row: row.hour, 1)
            data_stamp_t = df_stamp_t.drop(['date'], 1).values
        elif self.timeenc == 1:
#            data_stamp_t = time_features(pd.to_datetime(df_stamp_t.values), freq=self.freq)
            data_stamp_t = data_stamp_t.transpose(1, 0)
            data_stamp_t = df_stamp_t.values

        self.data_x = data[border1:border2]
        self.data_y = data_t[border1_v:border2_v]
        self.data_stamp = data_stamp[border1:border2]
        self.data_stamp_t = data_stamp_t[border1_v:border2_v]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        seq_x = self.data_x[s_begin:s_end]
        # set lag seq_x
        seq_x = self.set_lag_missing(seq_x, self.var_info,'M').values

        r_begin = s_begin * self.pred_len
        r_end = r_begin + self.seq_len*self.pred_len
        seq_y = self.data_y[r_end-self.pred_len:r_end].values

        # time feagure index       
        # seq_x_mark = self.data_stamp[s_begin:s_end].values
        # seq_y_mark = self.data_stamp_t[r_end-1:r_end].values
        
        return seq_x, seq_y #seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, features='MS', data_path='custom.csv',
                 target='GDP', scale=True, inverse=False, timeenc=0, freq='m', cols=None):
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        self.period = {'M': ['2000-01','2023-06'], 'Q':['2000-01','2023-06']}
        self.start_M = self.period['M'][0]
        self.end_M = self.period['M'][1]
        self.start_Q = self.period['Q'][0]
        self.end_Q = self.period['Q'][1]

#        self.load_data_timeindex = load_data_timeindex
#        self.set_lag_missing = set_lag_missing
#        self.repeat_label_row = repeat_label_row        
        
        self.__read_data__()

    def __read_data__(self):
        self.scaler_m = MinMaxScaler()
        # self.scaler_q = MinMaxScaler()
        
        path = os.path.join(self.root_path, self.data_path)
        df_Q, _, df_M, _, self.var_info = self.load_data_timeindex(path)
        
        cols_Q = list(df_Q.columns)
        cols_Q.remove(self.target)
        df_Q = df_Q[cols_Q + [self.target]]
        
        df_M = df_M.loc[self.start_M:]
        df_Q = df_Q.loc[self.start_Q:]
        
        # temp Q_variable insert
        df_M = pd.concat([df_M,df_Q],axis=1)
        cols_M = list(df_M.columns)

        df_Q = self.repeat_label_row(df=df_Q,pred_len=self.pred_len,repeat=3)
        
        border1 = len(df_M) - self.seq_len
        border2 = len(df_M)

        if self.features == 'M' or self.features == 'MS':
            df_data = df_M[cols_M]
        elif self.features == 'S':
            df_data = df_M[[self.target]]

        if self.scale:
            # self.scaler_m.fit(df_data.values)
            # data = self.scaler.transform(df_data.values)
            df_data_cols = df_data.columns
            df_data_index = df_data.index
            self.scaler_m.fit(df_data.values)
            data = self.scaler_m.fit_transform(df_data.values)
            data = df_data.values
            data = pd.DataFrame(data,columns=df_data_cols, index=df_data_index)
            # data_t = df_data_t # pd.DataFrame(data_t,columns=df_data_t_cols, index=df_data_t_index)
        else:
            data = df_data.values
            # data_t = df_data_t.values
            
        tmp_stamp = df_data.index
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(
                lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
#        elif self.timeenc == 1:
#            data_stamp = time_features(pd.to_datetime(
#                df_stamp['date'].values), freq=self.freq)
#            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse: 
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
