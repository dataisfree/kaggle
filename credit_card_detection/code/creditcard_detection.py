# -*- coding: utf-8 -*-
"""
classification
"""

import os
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE

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
print(df.head(2))

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
y = train[feature_name]

# 样本平衡处理
y_cnt_ = Counter(y.values)
print(dir(y_cnt_.keys()))
class_num = len(y_cnt_.keys())
class_0_rate = round(int(y_cnt_.get(0)) / sum(y_cnt_.values()), 4)
class_1_rate = round(int(y_cnt_.get(1)) / sum(y_cnt_.values()), 4)
over_sampling_trainX, over_sampling_trainy = dc.imbalance(
	trainX=X, trainy=y, class_num=class_num, positive_negative_perc=class_1_rate
)
'''
注：imbalance返回np.adarray, 后续需要用到pd.dataframe, 考虑在何处完成df格式的完成
'''

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
trainy = y.copy()

# xgb Dmatrix


# train


# validation
