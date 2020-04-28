import pandas as pd
import numpy as np

data_all = pd.read_csv('data/data_all_raw.csv')
data_all = data_all.reset_index(drop=True).drop(['Unnamed: 0'], axis=1).drop(['index'], axis=1)
print(data_all.shape)
data_all = data_all[data_all['username'].notna()]   # Drop columns without a username
print(data_all.shape)
data_all = data_all[data_all['text'].notna()]       # Drop empty tweets
print(data_all.shape)
data_all = data_all[data_all['text'].notna()]       # Drop empty tweets
print(data_all.shape)
data_all = data_all[data_all['Followers'] != 0]     # Drop inactive accounts (no followers)
print(data_all.shape)
data_all = data_all[[col for col in data_all if col not in ['age']] + ['age']]
print(data_all.head())
print(data_all.shape)

data_all.to_csv('data/data_raw_clean.csv')