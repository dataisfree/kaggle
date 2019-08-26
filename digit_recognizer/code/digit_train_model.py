# -*- coding: utf-8 -*-
"""
name: 手写字体识别验证

creater: chenzhiwei

datetime: 2019-07-11
"""

import os
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

base_path = os.getcwd()

file_path = '../source_data/train.csv'
df = pd.read_csv(file_path)
test_file_path = '../source_data/test.csv'
test_df = pd.read_csv(test_file_path)

# X_train = df[:1000].iloc[:, 1:df.shape[1]]
# y_train = df[:1000].iloc[:, 0]
# X_test = test_df[:100]

df_index = df.index.values
np.random.shuffle(df_index)
X_train_index = df_index[: 960]
X_test_index = df_index[960:1200]

X_train = df.ix[X_train_index, 1:df.shape[1]]
y_train = df.ix[X_train_index, [0]]
X_test = df.ix[X_test_index, 1:df.shape[1]]
y_test = df.ix[X_test_index, [0]]

print X_train.head(), y_train.head()

# train data
'''
xgboost从pandas.DataFrame中数据生成xgboost可用的DMatrix对象
To load a Pandas data frame into DMatrix:
------------------------------------------------------------------------------
data = pandas.DataFrame(np.arange(12).reshape((4,3)), columns=['a', 'b', 'c'])
label = pandas.DataFrame(np.random.randint(2, size=4))
dtrain = xgb.DMatrix(data, label=label)

------------------------------------------------------------------------------
'''

dtrain = xgb.DMatrix(data=X_train, label=y_train)
dtest = xgb.DMatrix(data=X_test, label=y_test)

params = {'max_depth': 5, 'eta': 0.1, 'objective': 'multi:softmax',
		  'eval_metric': 'merror', 'verbosity': 0, 'num_class': 10}

evals_result = {}
watchlist = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=200,
				  evals=watchlist, evals_result=evals_result)

y_pred = model.predict(dtest)
y_pred_margin = model.predict(dtest, output_margin=True)
print classification_report(y_true=dtest.get_label(), y_pred=y_pred)
print '-.'*30
print confusion_matrix(y_true=dtest.get_label(), y_pred=y_pred)

model.get_dump()
print model.get_dump()

### save model
# after training, the model can be saved
model.save_model('../model/xgboost_digit_image.model')
# also, model and its feature map can be dumped to a text file
model.dump_model('../model/dump.raw.txt')
# also, can use pickle package to save model in disk
pickle.dump(model, open('../model/test.pkl', 'wb'))

### load model
# a saved can be loaded
load_model = xgb.Booster({'nthread': 4})	# init model
load_model.load_model(r'../model/xgboost_digit_image.model')

load_model_pkl = pickle.load(open('../model/test.pkl', 'rb'))


### predict
dpredict = xgb.DMatrix(data=test_df[:1000])
pred = model.predict(dpredict)
print test_df[:1000].index.values
imageId = pd.DataFrame(data=test_df[:1000].index.values, columns=['ImageId'])
label = pd.DataFrame(data=pred, columns=['Label'], dtype=int)

result = pd.concat([imageId, label], axis=1)
print result.head()

xgb.plot_tree(model, num_trees=0)

print '--test--'*5
pred_1 = load_model.predict(dpredict)
pred_2 = load_model_pkl.predict(dpredict)
print confusion_matrix(pred, pred_1)
print confusion_matrix(pred, pred_2)

'''
ImageId,Label
1,0
2,0
3,0
'''

print(12)

