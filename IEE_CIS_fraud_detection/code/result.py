# -*- coding: utf-8

import pandas as pd
import time

df = pd.read_csv('../predict/predict_result_train_4w.csv', chunksize=10000)
df = df.get_chunk(200)
print(df.head())

for line in range(df.shape[0]):
	print('#  current line is: ')
	print(df.iloc[line, :])
	time.sleep(1)
	if line >= 100:
		break
print('run finish!')
