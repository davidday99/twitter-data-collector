import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_all = pd.read_csv('data/profile_data_all.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)

X = data_all.loc[:, 'handle':'avg_word_count']
Y = data_all.loc[:, 'age_group']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, stratify=Y, random_state=1)

df_train = pd.concat([X_train, Y_train], axis=1).reset_index(drop=True)
df_test = pd.concat([X_test, Y_test], axis=1).reset_index(drop=True)

print(df_train.head())
print(df_test.head())

df_train.to_csv('data/profile_data_train.csv')
df_test.to_csv('data/profile_data_test.csv')