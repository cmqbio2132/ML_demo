#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:50:04 2023

@author: yu
"""

import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt




df = sns.load_dataset("iris")

df_x = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
df_y = df["species"]

train_x, test_x,train_y, test_y = train_test_split(df_x, df_y, random_state=1)

model = tree.DecisionTreeClassifier(max_depth=8, random_state=1)
model.fit(train_x, train_y)
model.predict(test_x)

model.score(test_x, test_y)

plot_tree(model, feature_names=train_x.columns, class_names=True, filled=True)

plt.savefig(f"/Users/yu/Desktop/ml_training/dataset/iris/iris_depth_8.png",format = "png", dpi = 300)

