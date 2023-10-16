import torch
import pandas as pd
from sklearn import preprocessing as pp
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import numpy as np
import random
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
import time
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

pd.set_option('display.max_colwidth', None)
columns_name = ['user', 'item', 'ratings']
train_df = pd.read_csv("./Data/train.txt")
valid_df = pd.read_csv("./Data/valid.txt")
test_df = pd.read_csv("./Data/test.txt")

le_user = pp.LabelEncoder()
le_item = pp.LabelEncoder()
train_df['user_idx'] = le_user.fit_transform(train_df['user'].values)
train_df['item_idx'] = le_item.fit_transform(train_df['item'].values)
train_user = train_df['user'].unique()
train_item = train_df['item'].unique()

test_df = test_df[(test_df['user'].isin(train_user)) & (test_df['item'].isin(train_item))]
valid_df = valid_df[(valid_df['user'].isin(train_user)) & (valid_df['item'].isin(train_item))]

valid_df['user_idx'] = le_user.transform(valid_df['user'].values)
valid_df['item_idx'] = le_item.transform(valid_df['item'].values)
test_df['user_idx'] = le_user.transform(test_df['user'].values)
test_df['item_idx'] = le_item.transform(test_df['item'].values)

n_users = train_df['user_idx'].nunique()
n_items = train_df['item_idx'].nunique()
print("Number of Unique Users : ", n_users)
print("Number of unique Items : ", n_items)

latent_dim = 64
n_layers = 3


def convert_to_sparse_tensor(dok_mtrx):

    dok_mtrx_coo = dok_mtrx.tocoo().astype(np.float32)
    values = dok_mtrx_coo.data
    indices = np.vstack((dok_mtrx_coo.row, dok_mtrx_coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = dok_mtrx_coo.shape

    dok_mtrx_sparse_tensor = torch.sparse.FloatTensor(i, v, torch.Size(shape))

    return dok_mtrx_sparse_tensor
