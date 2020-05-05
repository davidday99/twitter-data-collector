import random
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn import preprocessing
import lightgbm as lgb
from catboost import Pool, CatBoostClassifier
# from fastai.tabular import *
from sklearn.dummy import DummyClassifier



df_train = pd.read_csv('data/train.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/test.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
X_train = df_train.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop('tweets_text', axis=1)
Y_train = df_train.age_group.values
X_test = df_test.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop('tweets_text', axis=1)
Y_test = df_test.age_group.values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)


important_embeddings = {547, 548, 611, 448, 335, 561, 727, 274, 110, 124, 355, 438, 423, 629, 354, 165, 160, 518}
for i in range(768):
    if i not in important_embeddings:
        name = 'embed' + str(i)
        X_train = X_train.drop(name, axis=1)
        X_test = X_test.drop(name, axis=1)

print(X_train.head())


# ~57%
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)
rf = RandomForestClassifier(n_estimators=2000, min_samples_split=2, min_samples_leaf=1, max_depth=50, bootstrap=False, random_state=0)
rf.fit(X_train, Y_train)

score = rf.score(X_test, Y_test)
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt

predictions = rf.predict(X_test)
cm = confusion_matrix(Y_test, predictions, normalize='true')
df_cm = pd.DataFrame(cm, range(4), range(4))
sn.heatmap(df_cm, annot=True)
plt.ylabel('True')
plt.xlabel('Predicted')
plt.show()


# Ol' reliable, 59.75
model = CatBoostClassifier(iterations=5000,
                           learning_rate=.02,
                           depth=4,
                           loss_function='MultiClass')
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))


# Ol' Reliable 1: 59.5%
final_xgb = XGBClassifier(learning_rate=0.05, n_estimators=600, max_depth=4, seed=27, objective= 'multi:softmax',
                          nthread=4)

final_xgb.fit(X_train, Y_train)
print(final_xgb.score(X_test, Y_test))

# Ol' Reliable: 59.5%
final_xgb = XGBClassifier(learning_rate =0.1, n_estimators=820, max_depth=25,
                         min_child_weight=1, gamma=0, subsample=0.9, colsample_bytree=0.1,
                         objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27)

final_xgb.fit(X_train, Y_train)
print(final_xgb.score(X_test, Y_test))

model = CatBoostClassifier(iterations=450,
                           learning_rate=.02,
                           depth=5,
                           loss_function='MultiClass')
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))



