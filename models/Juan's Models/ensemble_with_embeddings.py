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
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report


# Note to future: models achieve 55-58% accuracy

df_train = pd.read_csv('data/train.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
df_test = pd.read_csv('data/test.csv').drop('Unnamed: 0', axis=1).reset_index(drop=True)
X_train = df_train.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop('tweets_text', axis=1).drop("Logit0", axis=1).drop("Logit1", axis=1).drop("Logit2", axis=1).drop("Logit3", axis=1)
Y_train = df_train.age_group.values
X_test = df_test.copy(deep=True).drop('age_group', axis=1).drop('hashtags', axis=1).drop("handle", axis=1).drop('tweets_text', axis=1).drop("Logit0", axis=1).drop("Logit1", axis=1).drop("Logit2", axis=1).drop("Logit3", axis=1)
Y_test = df_test.age_group.values
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# .drop("Logit0", axis=1).drop("Logit1", axis=1).drop("Logit2", axis=1).drop("Logit3", axis=1)

# Dummy as benchmark
# dummy_clf = DummyClassifier(strategy='stratified')# # Best catboost, ~
# # X_train["contains_profanity"] = X_train["contains_profanity"].astype(np.int)
# # X_test["contains_profanity"] = X_test["contains_profanity"].astype(np.int)
# # cat_features = ['contains_profanity']
# # train_dataset = Pool(data=X_train,
# #                      label=Y_train,
# #                      cat_features=cat_features)
# # eval_dataset = Pool(data=X_test,
# #                     label=Y_test,
# #                     cat_features=cat_features)
# # model = CatBoostClassifier(iterations=300,
# #                            depth=5,
# #                            learning_rate=.2,
# #                            loss_function='MultiClass')
# # model.fit(train_dataset)
# # print(model.score(eval_dataset))
# # cat_feat_imps = model.get_feature_importance()
# # importances = list(cat_feat_imps)
# # feature_importances = [(feature, round(importance, 4)) for feature, importance in zip(X_train.columns, importances)]
# # #[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]
# # feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# # print(len(feature_importances))
# # [print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances]
# # plt.bar(range(len(importances)), importances)
# # plt.show()
# dummy_clf.fit(X_train, Y_train)
# print(dummy_clf.score(X_test, Y_test))



# Best RF: ~54%
rf = RandomForestClassifier(n_estimators=1000, min_samples_split=5, min_samples_leaf=1, max_features='auto', max_depth=500, bootstrap=False, random_state=0)
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

# Best XGB: ~53%
final_xgb = XGBClassifier(learning_rate =0.01, n_estimators=1000, max_depth=6,
                         min_child_weight=5, gamma=0, subsample=0.7, colsample_bytree=0.3,
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


# Best logistic, ~53%
model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=100000, penalty='l1', C=0.03, random_state=0)
model.fit(X_train, Y_train)
print(model.score(X_test, Y_test))

















# X_train["contains_profanity"] = X_train["contains_profanity"].astype(np.int)
# X_test["contains_profanity"] = X_test["contains_profanity"].astype(np.int)
# #
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

# param_test1 = {
#  'max_depth':[7, 8, 9, 10],
#  'min_child_weight':[2, 3, 4],
# }
#
# gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=140, max_depth=5,
#  min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
#  objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27),
#  param_grid = param_test1, n_jobs=4,iid=False, cv=5, verbose=2)
# gsearch1.fit(X_train, Y_train)
# print(gsearch1.best_params_, gsearch1.best_score_)
#
# final_xgb = XGBClassifier(learning_rate =0.05, n_estimators=450, max_depth=1,
#                          min_child_weight=4, gamma=0, subsample=0.9, colsample_bytree=0.1,
#                          objective= 'multi:softmax', nthread=4, scale_pos_weight=1, seed=27)
#
# final_xgb.fit(X_train, Y_train)
# print(final_xgb.score(X_test, Y_test))
#
#
# X_train = X_train.fillna(0)
# X_test = X_test.fillna(0)
#
# rf = RandomForestClassifier(n_estimators=1000, max_depth=10, min_samples_split=2, min_samples_leaf=5)
#
# rf.fit(X_train, Y_train)
# print(rf.score(X_test, Y_test))

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# clf = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=10000)
# grid_values = {'penalty': ['l1'],'C':[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]}
# grid_clf = GridSearchCV(clf, param_grid=grid_values, verbose=1, n_jobs=-1)
# grid_clf.fit(X_train, Y_train)
# print(grid_clf.best_params_, grid_clf.best_score_)
# print(grid_clf.score(X_test, Y_test))

# model = LogisticRegression(solver='liblinear', multi_class='ovr', max_iter=1000, penalty='l1', C=0.5)
# model.fit(X_train, Y_train)
# coef = np.array(model.coef_)
# print(coef.shape)
# print(coef[0].shape)
#
# for c in coef:
#     for i in range(len(c)):
#         if abs(c[i]) > 0.1:
#             print(X_train.columns[i], c[i])
#
# plt.bar(X_train.columns, coef[0])
# plt.show()

# clf = svm.SVC(C=5, kernel='poly', degree=10)
# clf.fit(X_train, Y_train)
# print(clf.score(X_test, Y_test))