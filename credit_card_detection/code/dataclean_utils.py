# -*- coding: utf-8 -*-
"""
数据清洗脚本
"""
from collections import Counter
from imblearn.over_sampling import SMOTE

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
		result_dict[dis_name] = Counter(temp_data[dis_name].values).most_common(1)[0][0]
	# result_dict[dis_name] = temp_data[dis_name].value_counts().index.values[0]
	for con_name in con_names:
		result_dict[con_name] = temp_data[con_name].values.mean().round(4)
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
	temp_trainX = trainX.copy()
	temp_trainy = trainy.copy()

	# sm = SMOTE(sampling_strategy='minority', random_state=7)
	sm = SMOTE(ratio='minority', random_state=7)
	oversampling_trainX, oversampling_trainy = sm.fit_resample(X=temp_trainX, y=temp_trainy)
	return oversampling_trainX, oversampling_trainy


def imbalance(trainX, trainy, unique_class_val=2, positive_negative_perc=0.5):
	temp_trainX = trainX.copy()
	temp_trainy = trainy.copy().astype('int')

	if unique_class_val < 3:
		if positive_negative_perc < 0.4 or (1 - positive_negative_perc) < 0.4:
			return dive_imbalance_data(trainX=temp_trainX, trainy=temp_trainy)
		else:
			print(u'当前数据集暂不支持进行“样本均衡处理”！')
			return temp_trainX, temp_trainy
	else:
		print(u'当前数据集暂不支持进行“样本均衡处理”！')
		return temp_trainX, temp_trainy
