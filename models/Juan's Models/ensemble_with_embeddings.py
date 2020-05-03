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

# Note to future: models achieve 55-58% accuracy

df_train = pd.read_csv('data/train_w_logits_and_embeddings.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/test_w_logits_and_embeddings.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
X_train = df_train.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop("tweets_text", axis=1)
Y_train = df_train.age_group.values
X_test = df_test.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop("tweets_text", axis=1)
Y_test = df_test.age_group.values

# # Dummy as benchmark
# dummy_clf = DummyClassifier(strategy="most_frequent")
# dummy_clf.fit(X_train, Y_train)
# print(dummy_clf.score(X_test, Y_test))
# X_train["contains_profanity"] = X_train["contains_profanity"].astype(np.int)
# X_test["contains_profanity"] = X_test["contains_profanity"].astype(np.int)
#
# cat_features = ['contains_profanity']
#
# train_dataset = Pool(data=X_train,
#                      label=Y_train,
#                      cat_features=cat_features)
#
# eval_dataset = Pool(data=X_test,
#                     label=Y_test,
#                     cat_features=cat_features)
#
# model = CatBoostClassifier(iterations=10,
#                            depth=10,
#                            learning_rate=0.1,
#                            loss_function='MultiClass')
# model.fit(train_dataset)
# print(model.score(eval_dataset))

param_test1 = {
 'max_depth':[7, 8, 9, 10],
 'min_child_weight':[2, 3, 4],
}

gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
 param_grid = param_test1, n_jobs=4,iid=False, cv=5, verbose=2)
gsearch1.fit(X_train, Y_train)
print(gsearch1.best_params_, gsearch1.best_score_)

final_xgb = XGBClassifier(learning_rate =0.05, n_estimators=450, max_depth=1,
                         min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.1,
                         objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27)

final_xgb.fit(X_train, Y_train)
print(final_xgb.score(X_test, Y_test))


X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

rf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2, min_samples_leaf=5)

rf.fit(X_train, Y_train)
print(rf.score(X_test, Y_test))
