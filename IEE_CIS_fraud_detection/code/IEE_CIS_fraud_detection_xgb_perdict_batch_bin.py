# -*- coding: utf-8 -*-
"""


"""

import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
import math
from collections import Counter

train_categorical_feature_names = [
	'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'M1',
	'M2',
	'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'have_p_emaildomain', 'have_r_emaildomain'
]
train_continuous_feature_names = [
	'TransactionDT',
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
# 部分特征离散化
special_cols_by_split = [
	'card1', 'card2', 'card3', 'card5', 'addr1', 'addr2',
]  # 待分箱的字段名
# load args
args_feature_names = []
with open('../model/args_cleaned_feature_names.txt', 'r+') as inf:
	for name in inf.readlines():
		args_feature_names.append(name.strip())
args_normal_cfeatures = pd.read_csv('../model/args_normalized_continuous_features.csv')
args_missing_value = pd.read_json('../model/args_missing_value_fill.json')
args_missing_value_dict = eval(args_missing_value.to_json())['args']
'''
pd.read_json对复杂结构的json会无能为力
'''
# args_split_bin_range = pd.read_json('../model/args_split_box_range.json')
with open('../model/args_split_box_range.json', 'r+') as inf:
	args_split_bin_range = inf.readlines()

args_split_bin_range_dict = eval(args_split_bin_range[0])
a = pd.io.json.json_normalize(json.loads(args_split_bin_range[0]))
b = json.loads(args_split_bin_range[0])
# print(args_split_bin_range)
# print('\n', pd.io.json.json_normalize(json.loads(args_split_bin_range[0])), '\n', json.loads(args_split_bin_range[0]))

# load data
df = pd.read_csv('../predict/test_transaction.csv')


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


def z_score(x, a_mean, a_std):
	if a_std == 0:
		return 0
	else:
		return (x - a_mean) / a_std


def math_ceil(x):
	return int(math.ceil(x))


def split_bin(x, range_index_dict=None):
	print('# cell x value is: ', x)
	if x == range_index_dict['fillna_value'] and isinstance(x, str):
		# 处理字符串型（表示其他类别）的cell值
		print('# cell x bin_index with fillna_value!', '\n', '#\tcondition: cell x == fillna_value!')
		return range_index_dict['fillna_value']
	if not (isinstance(x, float) or isinstance(x, int)):
		print('# cell x bin_index with fillna_value!', '\n', '#\tcondition: cell x not isinstance float or int!')
		return range_index_dict['fillna_value']

	cnt = 0
	for key in range_index_dict.keys():
		if key != 'fillna_value':
			key_value = range_index_dict[key].split(':')[-1]
			bin_low = key_value.split()[0].split('(')[-1]
			bin_high = key_value.split()[-1].split(')')[0]
			if float(bin_low) < float(x) <= float(bin_high):
				cnt += 1
				print('cell x in range: {}'.format(key_value))
				return key
			else:
				print('cell x not in range: {}'.format(key_value))

	if cnt == 0:
		raise ValueError('current feature split bin failed!, please check the values....')


result = pd.concat(
	[pd.Series([], name='TransactionID', dtype=int).reset_index(),
	 pd.Series([], name='isFraud').reset_index(drop=True)], axis=1
)

# load model
model = xgb.Booster()
model.load_model('../model/IEE_CIS_fraud_detection_cv_V1.model')

for i in range(df.shape[0]):
	if i % 2000 == 0:
		print('# current predict lines is: {low} - {high}'.format(low=i + 1, high=i + 2000))
		temp_df = df.ix[i:i + 1999, :].copy()
		print(temp_df.shape)
		data_template = np.zeros((temp_df.shape[0], len(args_feature_names)))
		data_template = pd.DataFrame(data_template, columns=args_feature_names, dtype=int)
		print(temp_df['P_emaildomain'].unique())
		temp_df['have_p_emaildomain'] = temp_df['P_emaildomain'].apply(lambda x: have_value(x))
		print(temp_df['R_emaildomain'].unique())
		temp_df['have_r_emaildomain'] = temp_df['R_emaildomain'].apply(lambda x: have_value(x))
		temp_df = temp_df.fillna(args_missing_value_dict)
		# 处理连续值
		for con_name in train_continuous_feature_names:
			mean = args_normal_cfeatures.ix[args_normal_cfeatures['Unnamed: 0'] == con_name, 'mean'].values.min()
			std = args_normal_cfeatures.ix[args_normal_cfeatures['Unnamed: 0'] == con_name, 'std'].values.min()
			data_template[con_name] = temp_df[con_name].apply(lambda x: z_score(x, a_mean=mean, a_std=std)).values
			if con_name.startswith('C'):
				data_template[con_name] = data_template[con_name].apply(lambda x: math_ceil(x)).values

		# 变量分箱(这里处理的全是离散变量)
		for special_col in special_cols_by_split:
			temp_df[special_col] = temp_df[special_col].apply(lambda x: split_bin(
				x, range_index_dict=args_split_bin_range_dict[special_col])).values

		# 处理离散值
		for cat_name in train_categorical_feature_names:
			temp_df[cat_name] = temp_df[cat_name].astype(str)
			uni_values = temp_df[cat_name].unique().tolist()
			for uni_value in uni_values:
				data_template[str(cat_name) + '_' + str(uni_value)] = \
					[1 if val == uni_value else 0 for val in temp_df[cat_name]]

		print('#  data_template shape: ', data_template.shape)
		data_template = data_template[args_feature_names].copy()
		print('#  data_template shape: ', data_template.shape)
		data_template.sort_index(axis=1, inplace=True)

		dtest = xgb.DMatrix(data_template)
		y_pred = model.predict(dtest)

		batch_result = pd.concat(
			[
				temp_df['TransactionID'].reset_index(),
				pd.Series(y_pred, name='isFraud').reset_index(drop=True)
			], axis=1
		)
		result = pd.concat([result, batch_result], axis=0)
		y_class = [1 if val >= 0.5 else 0 for val in y_pred]
		print('# current batch 0-1 percent: {}'.format(Counter(y_class)))

result.to_csv('../predict/predict_result.csv', index=False)

print('run finish!')
