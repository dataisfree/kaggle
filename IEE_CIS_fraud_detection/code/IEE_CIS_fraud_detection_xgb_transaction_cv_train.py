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

pd.set_option('max_columns', 100)

df = pd.read_csv('../data/cleaned_dataset_transaction.csv')

all_names = df.columns.values.tolist()
y_name = 'isFraud'
X_names = all_names.copy()
X_names.remove(y_name)

X = df[X_names].copy()
y = df[[y_name]].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


"""
cv: 
1. 目标： 测试参数的所有组合，用最少的步骤找到足够好的解决方案，找到相对最优的模型

2. xgb.cv()的重点参数：params/dtrin/num_boost_round/seed/nfold/metrics
	params: 字典，生成树的相关参数，主要调整的参数均位于字典内
	num_boost_round: 迭代轮次
	seed: 随机种子，确保在每一次迭代中用到的都是同一样的折叠（folds），便于比较不同参数的得分（best_scores）
	dtrain: 训练集，cv中仅需要传入训练集不需要传入测试集，因为：交叉验证是将训练集折叠后迭代的取一份作为测试集
	metrics: 度量指标，回归的mae/分类的error等
	
3. 主要测试参数：num_boost_round/early_stopping_rounds/max_depth/min_child_weight/subsample/colsample_bytree/eta
	num_boost_round: 非param参数字典的元素，作为独立参数传递给xgb.train方法。表示：训练的轮次或要建立多少颗树.
	其最优值高度依赖于其他参数，理论上其他参数变化时要同时re-tuned.
	early_stopping_rounds: xgboost提供了一种好的方式：在训练时去寻找训练轮次的最佳值，由于树是按顺序构建的，
	而不是在开始时一次性全部生成，所以我们可以在每个步骤中测试我们的模型，看看添加一个新的树/轮是否可以提高性能。
	利用测试集和基于测试集的一个度量标准去计算每一个轮次（boost）的性能表现，如果性能表现在N个boost内都没有“提升”
	（既有更好的表现）（N值由early_stopping_rounds指定）时，停止训练并记录最佳训练轮次（boost）.
	更好表现的定义: 指标稳定在一个常量early_stop_round不在变化， or 
				 递减性指标（如error）递减到一定值A后突然上升且后续early_stop_round内都在上升或未出现比前述定值A小的值，V字型
				 递增型指标（如auc）递增到一定值B后突然下降且后续early_stop_round内都在下降或未出现比前述定值B大的值，倒V字型
	max_depth: 树的深度，有资料的另一个角度的解释：一颗树中对于最长的分支从根节点到叶片最多允许存在的节点数.
				更深层次的树可以通过添加更多的节点来建模更复杂的关系，但是随着深度增加，分割变得不那么相关，
				有时只是由于噪声，导致模型过度拟合
	min_child_weight: 子对象中所需的实例权重(hessian)的最小和。如果树分区步骤导致一个叶子节点的实例权重之和
						小于min_child_weight，那么构建过程将放弃进一步的分区；当weight=1时，表示样本的数量值。
						个人理解：树中生成节点时所需要的最小样本权重和若未达到最小权重和则停止继续生长。
						小的权重允许算法生成一个具有少量样本数的子节点，会导致树越来越复杂，同时可能会过拟合
	注： max_depth+min_child_weight控制树的复杂度
	
	subsample: 每一轮迭代时参与的样本量的百分比，默认值为1（代表所有样本均参与其中）
	
	colsample_bytree: 构建每一颗树时参加其中的特征的百分比，默认值为1（所有特征均参与其中）
	
	注： subsample+colsample_bytree: 控制每一轮迭代时参与构建树的样本和特征的抽样数量。区别于所有样本和所有特征全部参与，在每一轮迭代时，
	我们可以在基于稍微不同的数据构建一个树，这使得它不太可能过度适合单个示例或特性。防止过拟合。
	
	eta: 学习率， 它对应于每一轮后与特征相关的权重的收缩，换句话说，它定义了我们在每一步所做的“修正”的数量。实操种，低学习率会产生较为鲁棒的
	模型，相对应的小的学习率需要更多的时间去训练，有可能多次的训练仅能获得细微的性能提升
	
"""
# define params dictionary
params = {
	'max_depth': 5, 'eta': 0.1, 'objective': 'binary:logistic',
	'eval_metric': ['error', 'auc'], 'min_child_weight': 1,
	'verbosity': 0, 'subsample': 1, 'colsample_bytree': 1,
}

# cv tuning max_depth and min_child_weight
# step: 设定参数备选值
'''
策略:
最大最小值范围宽一些，范围内值间距离大一些；
缩小范围和间距；
'''
gridsearch_params = [
	(max_depth, min_child_weight)
	for max_depth in range(5, 7)
	for min_child_weight in range(5, 7)
]
# step run cv on each of params pairs
min_auc = float('inf')
best_params = None
for max_depth, min_child_weight in gridsearch_params:
	print('# current run cv with max_depth={a:d}, min_child_weigth={b:d}'.format(
		a=max_depth, b=min_child_weight
	))
	params['max_depth'] = max_depth
	params['min_child_weight'] = min_child_weight
	num_boost = 999
	early_stop_boost = 10
	cv_results = xgb.cv(
		params=params,
		dtrain=dtrain,
		num_boost_round=num_boost,
		nfold=5,
		early_stopping_rounds=early_stop_boost,
		metrics=['error', 'auc'],
		seed=0
	)
	# print('# cv results is: ', cv_results)

	mean_auc = cv_results['test-auc-mean'].max()
	boost_rounds = cv_results['test-auc-mean'].argmax()
	print('\tauc {auc:.4f} for {round:d} rounds'.format(
		auc=mean_auc, round=boost_rounds
	))
	if mean_auc < min_auc:
		min_auc = mean_auc
		best_params = (max_depth, min_child_weight)
	print(1)
# step print best values
print('# current params: {}, {}, auc: {}'.format(
	best_params[0], best_params[1], min_auc
))
print('# current params: {}, {}, auc: {}'.format(
	*best_params, min_auc
))


params['max_depth'] = best_params[0]
params['min_child_weight'] = best_params[1]


# cv tuning subsample and colsample_bytree
'''
策略：
	选定候选百分比值；
	先从大数值组开始迭代，之后再迭代小数值组（由大到小）
'''
# step: 设定备选参数
gridsearch_params = [
	(subsample, colsample_bytree)
	for subsample in [i/10. for i in range(7, 9)]
	for colsample_bytree in [i/10. for i in range(7, 9)]
]

# step: run cv on each of params pairs
min_auc = float('inf')
best_params_samples = None
for subsample, colsample_bytree in reversed(gridsearch_params):
	print('# current run cv with subsample={}, colsample_bytree={}'.format(
		subsample, colsample_bytree
	))
	params['subsample'] = subsample
	params['colsample_bytree'] = colsample_bytree
	num_boost = 999
	early_stop_boost = 10

	cv_results_sample = xgb.cv(
		params=params,
		dtrain=dtrain,
		num_boost_round=num_boost,
		early_stopping_rounds=early_stop_boost,
		seed=0,
		nfold=5,
		metrics=['error', 'auc'],
	)
	# print('# cv result is: ', cv_results_sample)
	mean_auc = cv_results_sample['test-auc-mean'].max()
	boost_rounds = cv_results_sample['test-auc-mean'].argmax()
	print('\tbest AUC {:.6f} for {:d} rounds'.format(mean_auc, boost_rounds))
	if mean_auc < min_auc:
		min_auc = mean_auc
		best_params_samples = (subsample, colsample_bytree)
# step print the best values
print('# current params: {}, {}, AUC: {}'.format(
	*best_params_samples, min_auc
))
print('# current params: {}, {}, AUC: {}'.format(
	best_params_samples[0], best_params_samples[1], min_auc
))
params['subsample'] = best_params_samples[0]
params['colsample_bytree'] = best_params_samples[1]


# cv tuning eta
'''
策略：
	eta 由大到小
'''
# step 设置备选参数
gridsearch_params = [.3, .2, .1, .05, .01, .005]

# step run cv on each of parames pairs
min_auc = float('inf')
best_params_eta = None
for eta in gridsearch_params:
	print('# current run cv with eta={:.6f}'.format(eta))
	params['eta'] = eta

	num_boost = 999
	early_stop_boost = 10

	cv_results_eta = xgb.cv(
		params=params,
		dtrain=dtrain,
		num_boost_round=num_boost,
		early_stopping_rounds=early_stop_boost,
		seed=0,
		nfold=5,
		metrics=['error', 'auc'],
	)
	# print('# cv result is: {}'.format(cv_results_eta))
	mean_auc = cv_results_eta['test-auc-mean'].max()
	boost_rounds = cv_results_eta['test-auc-mean'].argmax()
	print('\tbest AUC {:.6f} for {:d} rounds'.format(mean_auc, boost_rounds))
	if mean_auc < min_auc:
		min_auc = mean_auc
		best_params_eta = (eta)
print('# current params: {}, AUC: {}'.format(
	best_params_eta[0], min_auc
))
params['eta'] = best_params_eta[0]


print('# after cv, the params: {}'.format(params))

# retrain model with best params
num_boost = 999
early_stop_boost = 10
watch_list = [(dtrain, 'train'), (dtest, 'val')]
val_results = {}
model = xgb.train(
	params=params,
	num_boost_round=num_boost,
	dtrain=dtrain,
	evals=watch_list,
	early_stopping_rounds=early_stop_boost,
	evals_result=val_results
)
print('# best auc: {:.6f} with {:d} rounds'.format(model.best_score, model.best_iteration+1))

# save model
model.save_model('../train.model')

# predict
'''
如果发生了early-stop，使用model.best_ntree_limit 来指定使用最优的结构树来进行预测。
model.predict(dtest, ntree_limit=model.best_ntree_limit)
'''
# case: 在训练脚本里预测
dtest = xgb.DMatrix(X_test)
pre = model.predict(dtest, ntree_limit=model.best_ntree_limit)
# case: 在新起任务里预测
pre_model = xgb.Booster()
pre_model.load_model('../train.model')
new_pre = pre_model.predict(dtest, ntree_limit=pre_model.best_ntree_limit)
