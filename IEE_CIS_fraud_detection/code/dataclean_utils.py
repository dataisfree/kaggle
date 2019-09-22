# -*- coding: utf-8 -*-
"""
数据清洗脚本
"""
from collections import Counter
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd


def create_missing_value_dict(data, dis_names, con_names):
	"""
	缺失值填充字典
	------------
	:param data:
	:param dis_names:	list, list_list 离散属性
	:param con_names:	list, like_list 连续属性
	:return:	dict <feature_name: value>
	"""
	temp_data = data.copy()
	result_dict = dict()
	if temp_data is None:
		raise IOError('missing dict: input dataset is None.')
	# 离散变量
	for dis_name in dis_names:
		y_cnt = Counter(temp_data[dis_name].values)
		if len(y_cnt) <= 1:
			if str(y_cnt.most_common(1)[0][0]) == 'nan':
				result_dict[dis_name] = 0
			else:
				result_dict[dis_name] = y_cnt.most_common(1)[0][0]
		else:
			if str(y_cnt.most_common(1)[0][0]) == 'nan':
				result_dict[dis_name] = y_cnt.most_common(2)[1][0]
			else:
				result_dict[dis_name] = y_cnt.most_common(1)[0][0]

	# result_dict[dis_name] = temp_data[dis_name].value_counts().index.values[0]
	for con_name in con_names:
		result_dict[con_name] = round(temp_data[con_name].mean(), 4)
	# result_dict[con_name] = round(temp_data[con_name].mean(), 4)

	return result_dict


# Z-标准化
def z_score(x):
	"""
	Z标准化
	------
	:param x:
	:return:
	"""
	if x.std() == 0:
		return 0
	else:
		return (x - x.mean()) / x.std()


# data normalized
def data_normalized_continuous_features(data, col_names=None):
	"""
	连续特征标准化
	------------
	:param data:	pd.DataFrame
	:param col_names:	list or like_list
	:return:
	"""
	temp_data = data.copy()
	if data is None:
		raise IOError('data normalized: input dataset is None.')
	if col_names is None or len(col_names) == 0:
		return [], []
	# collect arguments
	args_df = temp_data[col_names].describe().ix[['mean', 'std'], :].T
	normal_data = temp_data[col_names].apply(lambda x: z_score(x))
	return normal_data, args_df


# data onehot
def data_onehot_discrete_features(data, dis_names=None):
	"""
	离散特征one-hot
	--------------
	:param data: pd.DataFrame
	:param dis_names: list or like-list
	:return:
	"""
	temp_data = data.copy()
	if data is None:
		raise IOError('data onehot: input dataset is None.')
	if dis_names is None or len(dis_names) == 0:
		return []
	temp_data_dis = temp_data[dis_names].astype('str')
	for dis_name in dis_names:
		uni_values = temp_data_dis[dis_name].unique().tolist()
		for uni_value in uni_values:
			temp_data_dis[str(dis_name) + '_' + str(uni_value)] = \
				[1 if val == uni_value else 0 for val in temp_data_dis[dis_name]]
	temp_data_dis.drop(dis_names, axis=1, inplace=True)
	return temp_data_dis


# data balanced
def dive_imbalance_data(trainX, trainy):
	"""
	不平衡集SMOTE处理
	-----------------

	:param trainX: pd.DataFrame
	:param trainy: pd.DataFrame
	:return: tuple(np.ndarray, np.ndarray, list, list)
	"""
	temp_trainX = trainX.copy()
	temp_trainy = trainy.copy()
	trainX_names = temp_trainX.columns.values.tolist()
	trainy_names = temp_trainy.columns.values.tolist()

	# sm = SMOTE(sampling_strategy='minority', random_state=7)
	sm = SMOTE(ratio='minority', random_state=7)
	oversampling_trainX, oversampling_trainy = sm.fit_resample(X=temp_trainX, y=temp_trainy)
	return oversampling_trainX, oversampling_trainy, trainX_names, trainy_names


def imbalance(trainX, trainy, class_num=2, positive_negative_perc=0.5):
	"""
	暂仅实现对二元分类的样本平衡
	----------------------------
	:param trainX: pd.DataFrame
	:param trainy: pd.DataFrame
	:param class_num: int
	:param positive_negative_perc: float, in [0, 1]
	:return: tuple(np.ndarray, np.ndarray, list, list)
	"""
	temp_trainX = trainX.copy()
	temp_trainy = trainy.copy().astype('int64')

	trainX_names = temp_trainX.columns.values.tolist()
	trainy_names = temp_trainy.columns.values.tolist()

	if 3 > class_num > 1:
		if positive_negative_perc < 0.4 or (1 - positive_negative_perc) < 0.4:
			return dive_imbalance_data(trainX=temp_trainX, trainy=temp_trainy)
		else:
			print(u'当前数据集暂不支持进行“样本均衡处理”！')
			return temp_trainX.values, temp_trainy.values, trainX_names, trainy_names
	else:
		print(u'当前数据集暂不支持进行“样本均衡处理”！')
		return temp_trainX.values, temp_trainy.values, trainX_names, trainy_names


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


def split_box_flag_and_desc(x, dividing_point=None, fillna_value=None):
	a = 0
	'''
	range(len(div_point)-1) 
	目的：因为在间隔区间首尾两端增加了正负无穷大，取range(len(div_point)-1)是为了在for和if时避免最后一个区间时下标越界，
	'''
	if not isinstance(x, type(fillna_value)):
		pass
	if isinstance(x, float) or isinstance(x, int):
		for index in range(len(dividing_point)-1):
			if dividing_point[index] < x <= dividing_point[index + 1]:
				a += 1
				return index
		if a == 0:
			raise ValueError('x(float or int) is not find box!')
	else:
		return fillna_value


# 变量分箱离散
def split_box(data, type='width', width_interval=500, fillna_dict=None):
	"""
	变量分箱
	-------
	:param width_interval:
	:param data: pd.Series, e.g. df['column']
	:param type: string, {frequency: 等频分箱, width: 等宽分箱}
	:return:
	"""
	temp_data = data.copy()
	if isinstance(fillna_dict[temp_data.name], str):
		temp_data = temp_data.replace(fillna_dict[temp_data.name], np.nan)

	if type == 'frequency':
		pass
	elif type == 'width':
		min_x = temp_data.min()
		max_x = temp_data.max()
		a = len(str(int(abs(temp_data.mean()))))
		if a == 1:
			width_interval = 5
		elif a in [2, 3]:
			width_interval = 10
		elif a > 3:
			width_interval = 100
		else:
			print('分箱间隔计算出错')
		# width_interval = 1*10**(len(str(int(abs(temp_data.mean()))))-1)

		if width_interval == 0:
			width_interval = 5

		n = int((max_x - min_x) * 1.0 / width_interval)
		div_point = [int(min_x + (i + 1) * width_interval) for i in range(n)]
		div_point.insert(0, float('-inf'))
		div_point.extend([float('inf')])
		interval_range = {
			index: 'range:({} < x <= {})'.format(div_point[index], div_point[index + 1])
			for index in range(len(div_point) - 1)
		}
		fillna_val = fillna_dict[temp_data.name]
		interval_range['fillna_value'] = fillna_val

		return data.apply(
			lambda x: split_box_flag_and_desc(
				x,
				dividing_point=div_point,
				fillna_value=fillna_val)
		), interval_range
	else:
		pass
