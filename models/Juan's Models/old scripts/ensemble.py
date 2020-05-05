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
from fastai.tabular import *
from sklearn.dummy import DummyClassifier
from matplotlib import pyplot as plt

# Note to future: models achieve 55-58% accuracy

df_train = pd.read_csv('data/train_w_logits.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/test_w_logits.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
X_train = df_train.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop("tweets_text", axis=1)
Y_train = df_train.age_group.values
X_test = df_test.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop("tweets_text", axis=1)
Y_test = df_test.age_group.values

# Dummy as benchmark
dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(X_train, Y_train)
print(dummy_clf.score(X_test, Y_test))
X_train["contains_profanity"] = X_train["contains_profanity"].astype(np.int)
X_test["contains_profanity"] = X_test["contains_profanity"].astype(np.int)

cat_features = ['contains_profanity']

train_dataset = Pool(data=X_train,
                     label=Y_train,
                     cat_features=cat_features)

eval_dataset = Pool(data=X_test,
                    label=Y_test,
                    cat_features=cat_features)

model = CatBoostClassifier(iterations=450,
                           learning_rate=.02,
                           depth=5,
                           loss_function='MultiClass')
model.fit(train_dataset)
print(model.score(eval_dataset))

cat_feat_imps = model.get_feature_importance()
importances = list(cat_feat_imps)
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X_train.columns, importances)]
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(len(feature_importances))
[print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances]
plt.bar(range(len(importances)), importances)
plt.show()


final_xgb = XGBClassifier(learning_rate =0.05, n_estimators=450, max_depth=1,
                         min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.1,
                         objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27)

final_xgb.fit(X_train, Y_train)
print(final_xgb.score(X_test, Y_test))
xgb_feat_imps = final_xgb.feature_importances_
importances = list(xgb_feat_imps)
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X_train.columns, importances)]
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(len(feature_importances))
[print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances]
plt.bar(range(len(importances)), importances)
plt.show()


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

rf = RandomForestClassifier(n_estimators=500, max_depth=15, min_samples_split=2, min_samples_leaf=5)

rf.fit(X_train, Y_train)
print(rf.score(X_test, Y_test))

rf_feat_imps = rf.feature_importances_
importances = list(rf_feat_imps)
feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X_train.columns, importances)]
#[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
print(len(feature_importances))
[print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances]
plt.bar(range(len(importances)), importances)
plt.show()


