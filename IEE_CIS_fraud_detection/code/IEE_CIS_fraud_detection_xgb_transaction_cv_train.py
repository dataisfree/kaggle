# -*- coding: utf-8 -*-
"""
cv + train
"""

import os
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report


df = pd.read_csv('../data/cleaned_dataset.csv')

all_names = df.columns.values.tolist()
y_name = 'isFraud'
X_names = all_names.copy()
X_names.remove(y_name)

X = df[X_names].copy()
y = df[[y_name]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
	'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic',
	'eval_metric': ['error', 'auc'],
	'verbosity': 0
}

val_results = {}
watch_list = [(dtrain, 'train'), (dtest, 'val')]
model = xgb.train(
	params=params, dtrain=dtrain,
	num_boost_round=200, evals=watch_list,
	evals_result=val_results, early_stopping_rounds=20
)

print(model.best_score)