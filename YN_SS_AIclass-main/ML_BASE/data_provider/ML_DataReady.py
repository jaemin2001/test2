
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd


def add_features(df, features):
    for feature in features:
        df_grouped = df.groupby("sequence")[feature]
        df_rolling = df_grouped.rolling(5, center=True)

        df[feature + "_lag1"] = df_grouped.shift(1)
        df[feature + "_diff1"] = df[feature] - df[feature + "_lag1"]
        df[feature + "_lag2"] = df_grouped.shift(2)
        df[feature + "_diff2"] = df[feature] - df[feature + "_lag2"]
        df[feature + "_roll_mean"] = df_rolling.mean().reset_index(0, drop=True)
        df[feature + "_roll_std"] = df_rolling.std().reset_index(0, drop=True)
    df.dropna(axis=0, inplace=True)
    return


'''
    Here we need code to take data 
    from sqlite and trun it into a dataframe
'''
class dataloader:
    pass

# # Code to temporarily load data
# train = pd.read_csv("./datasets/train.csv")
# test = pd.read_csv("./datasets/test.csv")
# train_labels = pd.read_csv("./datasets/train_labels.csv")

# train = train.set_index(["sequence", "subject", "step"])
# test = test.set_index(["sequence", "subject", "step"])


# # Code to check for null data
# print("Checking if there are any missing values:")
# print("Train: {}".format(train.isnull().sum().sum()))
# print("Test: {}".format(test.isnull().sum().sum()))

# features = ["sensor_{:02d}".format(i) for i in range(13)]
# add_features(train, features)
# add_features(test, features)
# train.head()

# input_size = train.shape[1]
# sequence_length = len(train.index.get_level_values(2).unique())

# # Scaling test and train
# scaler = StandardScaler()
# train = scaler.fit_transform(train)
# test = scaler.transform(test)

# # Reshaping:
# train = train.reshape(-1, sequence_length, input_size)
# test = test.reshape(-1, sequence_length, input_size)
# print("After Reshape")
# print("Shape of training set: {}".format(train.shape))
# print("Shape of test set: {}".format(test.shape))

# # Splitting train data set into train and validation sets
# # validation size is selected as 0.2
# t_X, v_X, t_y, v_y = train_test_split(train, train_labels.state, test_size=0.20,
#                                       shuffle=True, random_state=0)

# # Converting train, validation and test data into tensors
# train_X_tensor = torch.tensor(t_X).float()
# val_X_tensor = torch.tensor(v_X).float()
# test_tensor = torch.tensor(test).float()

# # Converting train and validation labels into tensors
# train_y_tensor = torch.tensor(t_y.values)
# val_y_tensor = torch.tensor(v_y.values)

# # Creating train and validation tensors
# train_tensor = TensorDataset(train_X_tensor, train_y_tensor)
# val_tensor = TensorDataset(val_X_tensor, val_y_tensor)

# # Defining the dataloaders
# dataloaders = dict()
# dataloaders["train"] = DataLoader(train_tensor, batch_size=64, shuffle=True)
# dataloaders["val"] = DataLoader(val_tensor, batch_size=32)
# dataloaders["test"] = DataLoader(test_tensor, batch_size=32)
# print("Dataloaders are created!")
