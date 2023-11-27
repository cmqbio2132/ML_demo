#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 17:23:11 2023

@author: yu
"""

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree

#データロード
df = sns.load_dataset("titanic")


#学習データの準備。
#性別とチケットクラス、運賃のみで生存予測を行う。
df_x = df[["sex", "pclass", "fare"]]
df_y = df["survived"]

#性別がstrなので、ダミー変数にする。
#ダミー変数にしたものがboolで返ってきているので、astype(int)でキャストする
df_x = pd.get_dummies(df_x, drop_first=True)
df_x["sex_male"] = df_x["sex_male"].astype(int)

#データをトレーニングデータとテストデータに分割する。
#引数のrandom_stateは乱数ジェネレータで、これにシードを渡すとシードに応じた乱数が生成されてデータが分割される。
train_x, test_x, train_y, test_y = train_test_split(df_x,df_y,random_state=1)

#Treeモデルの生成。多分、ここでは前に渡したものと同じシード値を渡さないといけないっぽい？
#max_depthはツリーの分岐回数。
model = tree.DecisionTreeClassifier(max_depth=2, random_state=1)

#学習開始
model.fit(train_x, train_y)

#予測開始
model.predict(test_x)

#予測精度の算出
model.score(test_x, test_y)

#Treeのプロットをして可視化してみる。
plot_tree(model, feature_names=train_x.columns, class_names=True, filled=True)
