# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams["axes.unicode_minus"] = False

# load_data
# df_identity = pd.read_csv('../data/train_identity.csv/train_identity.csv')
# df_transaction = pd.read_csv('../data/train_transaction.csv/train_transaction.csv')
#
# df = df_identity.join(
# 	df_transaction.set_index('TransactionID'),
# 	on='TransactionID',
# 	how='right',
# 	lsuffix='_identity',
# 	rsuffix='_transaction'
# )
#
# print('# identity shape: ', df_identity.shape)
# print('# transaction shape: ', df_transaction.shape)
# print('# df shape: ',  df.shape)
#
# df_col_names = df.columns.values.tolist()
# class_name = 'isFraud'
# print('# df dtypes:', df.dtypes)
# print(df.head(2))

df = pd.read_csv('../data/train_transaction.csv/join_identity_transaction.csv')


# 变量衍生
def have_value(x):
	x.replace(np.NaN, None, inplace=True)
	if x is None:
		return 0
	else:
		return 1


df['have_p_emaildomain'] = df['P_emaildomain'].apply(lambda x: have_value(x))
print(df[['have_p_emaildomain', 'P_emaildomain']])
print(1)
