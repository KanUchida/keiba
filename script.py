# coding: utf8

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

PATH_DATA = '../data/predict_1.csv'


# これにそったやリ方で、欠損値「-」を中央値で埋めて、
# http://aial.shiroyagi.co.jp/2016/03/randomforest-for-quick-analysis/
tmp_df = pd.read_csv(PATH_DATA)
df = tmp_df.dropna()

# 学習データ
x_train = df.loc[:9000, ['_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13', '_c14', '_c15', '_c16', '_c17', '_c18', 'jockey', '_c20']]
x_test = df.loc[9000:, ['_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13', '_c14', '_c15', '_c16', '_c17', '_c18', 'jockey', '_c20']]


""" 欠損のあるデータの確認
missing = x_train.copy()
missing = missing.apply(pd.isnull, axis=0)
print missing.groupby(['_c3', '_c4', '_c5', '_c6', '_c7', '_c8', '_c9', '_c10', '_c11', '_c12', '_c13', '_c14', '_c15', '_c16', '_c17', '_c18', 'jockey', '_c20']).sum()
"""

# 教師データ
y_train = df.loc[:9000, 'flag'] # something
y_test = df.loc[9000:, 'flag']  # something


# 欠損値の処理
# http://sinhrks.hatenablog.com/entry/2016/02/01/080859


# リストワイズ法（欠損値は除去）
"""
X_train = x_train.dropna()
X_test = x_test.dropna()
Y_train = y_train.dropna()
Y_test = y_test.dropna()
"""

print x_train.shape
print x_test.shape
print y_train.shape
print y_test.shape


r_forest = RandomForestRegressor(
    n_estimators=100,
    criterion='mse',
    random_state=1,
    n_jobs=1,
)

# データセットの中に [ - ] があって、そいつを処理できないから、そこを直してあげなきゃいけない
# 欠損値の処理みたいなのでぐぐれば何かしら出てくるはず
r_forest.fit(x_train, y_train)
y_traing_preq = r_forest.predict(x_train)
y_test_preq = r_forest.predict(x_test)

print y_train
print y_traing_preq

for i, v in zip(y_traing_preq, y_train):
    print i, v