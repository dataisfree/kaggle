# -*- coding: utf-8 -*-
"""

"""
import os
import numpy as np
import pandas as pd
import time
import math
from collections import Counter
import matplotlib as mpl
import json
import matplotlib.pyplot as plt
import dataclean_utils as dc
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import classification_report

mpl.rcParams['font.family'] = 'SimHei'
mpl.rcParams["axes.unicode_minus"] = False

# load_data
df_identity = pd.read_csv('../data/train_identity.csv')
df_transaction = pd.read_csv('../data/train_transaction.csv')

df = df_identity.join(
	df_transaction.set_index('TransactionID'),
	on='TransactionID',
	how='inner',
	lsuffix='_identity',
	rsuffix='_transaction'
)

print('# identity shape: ', df_identity.shape)
print('# transaction shape: ', df_transaction.shape)
print('# df shape: ', df.shape)

df_col_names = df.columns.values.tolist()
class_name = 'isFraud'
print('# df dtypes:', df.dtypes)
print(df.head(2))

# df = pd.read_csv('../data/join_identity_transaction.csv')

# df = df[:2000].copy()


# 变量衍生
def have_value(x):
	"""
	衍生变量是否有值
	----------------
	:param x:
	:return:
	"""
	if isinstance(x, np.float) and pd.isnull(x):
		return 0
	else:
		return 1


df['have_p_emaildomain'] = df['P_emaildomain'].apply(lambda x: have_value(x))
print(df['R_emaildomain'].unique())
df['have_r_emaildomain'] = df['R_emaildomain'].apply(lambda x: have_value(x))
print(df[['have_p_emaildomain', 'P_emaildomain']])
print(df[['have_r_emaildomain', 'R_emaildomain']])

train_feature_names = [
	'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'id_12', 'id_13',
	'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26',
	'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38',
	'DeviceType', 'isFraud', 'TransactionDT', 'TransactionAmt', 'ProductCD', 'card1', 'card2', 'card3',
	'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4',
	'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
	'D9',
	'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'V1', 'V2', 'V3',
	'V4',
	'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21',
	'V22',
	'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V38',
	'V39',
	'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51', 'V52', 'V53', 'V54', 'V55',
	'V56',
	'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68', 'V69', 'V70', 'V71', 'V72',
	'V73',
	'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85', 'V86', 'V87', 'V88', 'V89',
	'V90',
	'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101', 'V102', 'V103', 'V104', 'V105',
	'V106',
	'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115', 'V116', 'V117', 'V118', 'V119', 'V120',
	'V121',
	'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129', 'V130', 'V131', 'V132', 'V133', 'V134', 'V135',
	'V136',
	'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144', 'V145', 'V146', 'V147', 'V148', 'V149', 'V150',
	'V151',
	'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158', 'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165',
	'V166',
	'V167', 'V168', 'V169', 'V170', 'V171', 'V172', 'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180',
	'V181',
	'V182', 'V183', 'V184', 'V185', 'V186', 'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195',
	'V196',
	'V197', 'V198', 'V199', 'V200', 'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210',
	'V211',
	'V212', 'V213', 'V214', 'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225',
	'V226',
	'V227', 'V228', 'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240',
	'V241',
	'V242', 'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255',
	'V256',
	'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270',
	'V271',
	'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285',
	'V286',
	'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300',
	'V301',
	'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315',
	'V316',
	'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330',
	'V331',
	'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339', 'have_p_emaildomain', 'have_r_emaildomain'
]
train_categorical_feature_names = [
	'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24',
	'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37',
	'id_38', 'DeviceType', 'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'M1',
	'M2',
	'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'have_p_emaildomain', 'have_r_emaildomain'
]
train_continuous_feature_names = [
	'id_01', 'id_02', 'id_03', 'id_04', 'id_05', 'id_06', 'id_07', 'id_08', 'id_09', 'id_10', 'id_11', 'TransactionDT',
	'TransactionAmt', 'dist1', 'dist2', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12',
	'C13',
	'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'V1', 'V2',
	'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
	'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'V29', 'V30', 'V31', 'V32', 'V33', 'V34', 'V35',
	'V36', 'V37', 'V38', 'V39', 'V40', 'V41', 'V42', 'V43', 'V44', 'V45', 'V46', 'V47', 'V48', 'V49', 'V50', 'V51',
	'V52',
	'V53', 'V54', 'V55', 'V56', 'V57', 'V58', 'V59', 'V60', 'V61', 'V62', 'V63', 'V64', 'V65', 'V66', 'V67', 'V68',
	'V69',
	'V70', 'V71', 'V72', 'V73', 'V74', 'V75', 'V76', 'V77', 'V78', 'V79', 'V80', 'V81', 'V82', 'V83', 'V84', 'V85',
	'V86', 'V87', 'V88', 'V89', 'V90', 'V91', 'V92', 'V93', 'V94', 'V95', 'V96', 'V97', 'V98', 'V99', 'V100', 'V101',
	'V102', 'V103', 'V104', 'V105', 'V106', 'V107', 'V108', 'V109', 'V110', 'V111', 'V112', 'V113', 'V114', 'V115',
	'V116', 'V117', 'V118', 'V119', 'V120', 'V121', 'V122', 'V123', 'V124', 'V125', 'V126', 'V127', 'V128', 'V129',
	'V130',
	'V131', 'V132', 'V133', 'V134', 'V135', 'V136', 'V137', 'V138', 'V139', 'V140', 'V141', 'V142', 'V143', 'V144',
	'V145', 'V146', 'V147', 'V148', 'V149', 'V150', 'V151', 'V152', 'V153', 'V154', 'V155', 'V156', 'V157', 'V158',
	'V159', 'V160', 'V161', 'V162', 'V163', 'V164', 'V165', 'V166', 'V167', 'V168', 'V169', 'V170', 'V171', 'V172',
	'V173', 'V174', 'V175', 'V176', 'V177', 'V178', 'V179', 'V180', 'V181', 'V182', 'V183', 'V184', 'V185', 'V186',
	'V187', 'V188', 'V189', 'V190', 'V191', 'V192', 'V193', 'V194', 'V195', 'V196', 'V197', 'V198', 'V199', 'V200',
	'V201', 'V202', 'V203', 'V204', 'V205', 'V206', 'V207', 'V208', 'V209', 'V210', 'V211', 'V212', 'V213', 'V214',
	'V215', 'V216', 'V217', 'V218', 'V219', 'V220', 'V221', 'V222', 'V223', 'V224', 'V225', 'V226', 'V227', 'V228',
	'V229', 'V230', 'V231', 'V232', 'V233', 'V234', 'V235', 'V236', 'V237', 'V238', 'V239', 'V240', 'V241', 'V242',
	'V243', 'V244', 'V245', 'V246', 'V247', 'V248', 'V249', 'V250', 'V251', 'V252', 'V253', 'V254', 'V255', 'V256',
	'V257', 'V258', 'V259', 'V260', 'V261', 'V262', 'V263', 'V264', 'V265', 'V266', 'V267', 'V268', 'V269', 'V270',
	'V271',
	'V272', 'V273', 'V274', 'V275', 'V276', 'V277', 'V278', 'V279', 'V280', 'V281', 'V282', 'V283', 'V284', 'V285',
	'V286',
	'V287', 'V288', 'V289', 'V290', 'V291', 'V292', 'V293', 'V294', 'V295', 'V296', 'V297', 'V298', 'V299', 'V300',
	'V301',
	'V302', 'V303', 'V304', 'V305', 'V306', 'V307', 'V308', 'V309', 'V310', 'V311', 'V312', 'V313', 'V314', 'V315',
	'V316',
	'V317', 'V318', 'V319', 'V320', 'V321', 'V322', 'V323', 'V324', 'V325', 'V326', 'V327', 'V328', 'V329', 'V330',
	'V331',
	'V332', 'V333', 'V334', 'V335', 'V336', 'V337', 'V338', 'V339'
]
X_names = train_feature_names.copy()
X_names.remove(class_name)

train_data = df[train_feature_names].copy()
# train_data = train_data[:100000].copy()

train_data = train_data.sample(frac=1)
train_data = train_data[:40000].copy()
train_data_y = train_data[train_data['isFraud'] == 1]
train_data = pd.concat([train_data, train_data_y])
train_data.reset_index(inplace=True)

trainX = train_data[X_names].copy()
trainy = train_data[[class_name]].copy()

missing_fill_dict = dc.create_missing_value_dict(
	data=trainX,
	dis_names=train_categorical_feature_names,
	con_names=train_continuous_feature_names)

# 部分属性填充值替换为“其他”
special_names = [
	'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36',
	'id_37', 'id_38', 'DeviceType', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'M1',	'M2',
	'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'have_p_emaildomain', 'have_r_emaildomain'
]
for fname in special_names:
	missing_fill_dict[fname] = 'unknown'

# 计数型特征取整
CntNames = []
for CntName in train_continuous_feature_names:
	if CntName.startswith('C'):
		CntNames.append(CntName)
for CntName in CntNames:
	missing_fill_dict[CntName] = int(math.ceil(missing_fill_dict[CntName]))

for con_name in train_continuous_feature_names:
	if str(missing_fill_dict.get(con_name)) == 'nan':
		missing_fill_dict[con_name] = 0

for key_value in missing_fill_dict.items():
	print(key_value)

# 缺失值填充
trainX.fillna(missing_fill_dict, inplace=True)

trainX_onehot = dc.data_onehot_discrete_features(data=trainX, dis_names=train_categorical_feature_names)

trainX_normal, trainX_args_df = dc.data_normalized_continuous_features(
	data=trainX,
	col_names=train_continuous_feature_names)

trainX_transpose = pd.concat([pd.DataFrame(trainX_normal), pd.DataFrame(trainX_onehot)], axis=1)
trainX_transpose.sort_index(axis=1, inplace=True)

y_cnt = Counter(trainy.ix[:, 0].values)
class_num = len(y_cnt.keys())
class_0_rate = round(int(y_cnt.get(0)) / sum(y_cnt.values()), 4)
class_1_rate = round(int(y_cnt.get(1)) / sum(y_cnt.values()), 4)
for col in trainX_transpose.columns.values:
	print('# colname: ', col, '# value: ', trainX_transpose[col][trainX_transpose[col].isnull()])

over_sampling_trainX, over_sampling_trainy, trainX_names, trainy_names = dc.imbalance(
	trainX=trainX_transpose,
	trainy=trainy,
	class_num=2,
	positive_negative_perc=class_1_rate
)
over_sampling_trainX = pd.DataFrame(data=over_sampling_trainX, columns=trainX_names)
over_sampling_trainy = pd.DataFrame(data=over_sampling_trainy, columns=trainy_names)

# 导出清洗后的数据集
clean_dataset = pd.concat([over_sampling_trainX, over_sampling_trainy], axis=1)
clean_dataset.to_csv('../data/cleaned_dataset.csv', index=False)

X_train, X_test, y_train, y_test = train_test_split(
	over_sampling_trainX, over_sampling_trainy,
	test_size=0.3,
	random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
	'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic',
	'eval_metric': ['error', 'auc'],
	'verbosity': 0
}
val_results = {}
watch_list = [(dtrain, 'train'), (dtest, 'val')]
model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100, evals=watch_list, evals_result=val_results)

model.save_model('../model/IEE_CIS_fraud_detection_v1.model')

# predict
predictX = xgb.DMatrix(X_test)
y_pred = model.predict(predictX)

y_concate = pd.DataFrame(np.concatenate((y_test.values, y_pred.reshape(-1, 1)), axis=1), columns=['y_true', 'y_pred'])
y_concate.to_csv('../data/predict_result.csv')

y_class = [1 if val >= 0.5 else 0 for val in y_pred]
print(classification_report(y_test, y_class))
print(Counter(y_class))
print(1)

# 参数留存
# 标准化参数
trainX_args_df.to_csv('../model/args_normalized_continuous_features.csv')
# 缺失值参数
with open('../model/args_missing_value_fill.json', 'w+') as outf:
	outf.write(json.dumps({'args': missing_fill_dict}))

# 特征名
with open('../model/args_cleaned_feature_names.txt', 'w+') as outf:
	for name in over_sampling_trainX.columns.values.tolist():
		outf.write(name + '\n')

print(1)

