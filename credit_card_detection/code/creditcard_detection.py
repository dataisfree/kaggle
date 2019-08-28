# -*- coding: utf-8 -*-
"""
classification
"""

import os
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE

import xgboost as xgb

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['font.family'] = "SimHei"
mpl.rcParams["axes.unicode_minus"] = False

base_path = os.getcwd()
file_name = 'creditcard.csv'

# load data
df = pd.read_csv(r'../data/creditcardfraud/creditcard.csv')
print(df.head(2))