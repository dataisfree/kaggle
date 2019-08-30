# -*- coding: utf-8 -*-
"""
classification
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
import pickle
import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt
import dataclean_utils as dc

mpl.rcParams['font.family'] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

base_path = os.getcwd()
file_name = 'creditcard.csv'

# load data
df = pd.read_csv(r'../data/creditcardfraud/creditcard.csv')

# split train test
df = df.sample(frac=1, replace=False, random_state=0, axis=0)    # replace: 是否有放回抽样, true:有放回
train = df[:-5000]
test = df[-5000:-1]

# col_names
# all_names = train.columns.values.tolist()		# 所有特征全部用于训练集，部分用于训练集时手动指定如下示
all_names = [
	"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
	"V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Class"

]
dis_names = []
con_names = [
	"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10", "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18",
	"V19", "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]
feature_name = 'Class'
X_names = all_names.copy()
X_names.remove(feature_name)



# data split
X = train[X_names]
y = train[[feature_name]]		# pd.DataFrame
# y = train[feature_name]		# pd.Series
# 样本平衡处理
# 接受np.adarray, shape=(n,) 一行n列， pd.DataFrame[class].values需要reshape(1, -1)后再用
y_cnt_ = Counter(y.ix[:, 0].values)
class_num = len(y_cnt_.keys())
class_0_rate = round(int(y_cnt_.get(0)) / sum(y_cnt_.values()), 4)
class_1_rate = round(int(y_cnt_.get(1)) / sum(y_cnt_.values()), 4)
over_sampling_trainX, over_sampling_trainy, trainX_names, trainy_names = dc.imbalance(
	trainX=X, trainy=y, class_num=class_num, positive_negative_perc=class_1_rate
)
over_sampling_trainX = pd.DataFrame(data=over_sampling_trainX, columns=trainX_names)
over_sampling_trainy = pd.DataFrame(data=over_sampling_trainy, columns=trainy_names)

# 填充缺失值
fill_dict = dc.create_missing_value_dict(data=over_sampling_trainX, dis_names=dis_names, con_names=con_names)
over_sampling_trainX.fillna(value=fill_dict, inplace=True)

# 连续特征标准化
data_norml, data_args_df = dc.data_normalized_continuous_features(data=over_sampling_trainX, col_names=con_names)

# 离散特征onehot
data_onehot = dc.data_onehot_discrete_features(data=over_sampling_trainX, dis_names=dis_names)

# 训练集拼接
trainX = pd.concat([pd.DataFrame(data_norml), pd.DataFrame(data_onehot)], axis=1)
trainX.sort_index(axis=1, inplace=True)
trainy = over_sampling_trainy.copy()
X_train, X_test, y_train, y_test = train_test_split(trainX, trainy, test_size=0.25, random_state=0)

# xgb Dmatrix
dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

params = {
	'max_depth': 5, 'eta': 0.1, 'objective': 'binary:hinge',
	'eval_metric': ['error', 'auc'],
	'verbosity': 0
}
val_results = {}
watch_list = [(dtrain, 'train'), (dtest, 'val')]
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100, evals=watch_list, evals_result=val_results)

# validation
y_pred = model.predict(dtest)
# y_pred = [1 if value >= 0.5 else 0 for value in y_pred]

y_pred_margin = model.predict(data=dtest, output_margin=True)

print(classification_report(y_true=dtest.get_label(), y_pred=y_pred))

print(confusion_matrix(y_true=dtest.get_label(), y_pred=y_pred))

# save model
model.save_model('../model/creditcard_fraud_detection.model')
pickle.dump(model, open('../model/crditcard_fraud_detection.pkl', 'wb'))
