#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:31:47 2018

@author: arndt
"""

from os import chdir
chdir("/home/arndt/git-reps/hatespeech/")
from skfasttext import SimpleFastTextClassifier

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier

df=pd.read_csv("data/kaggle-data.csv")

# data preperation
df=df[["id","comment_text","toxic"]]
df["label"]="__label__not_toxic"
df.loc[df["toxic"]==1,"label"]="__label__toxic"
df["comment_text"]=df["comment_text"].apply(str.replace,args=("\n"," "))
df["comment_text"]=df["comment_text"].apply(str.replace,args=("\"",""))

df_x = df.iloc[:,1]
df_y = df.iloc[:,3]

x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)

clf=SimpleFastTextClassifier()
model = clf.fit(x_train, y_train)
print(model.score(x_test, y_test))
print(model.score(x_train, y_train)) #model overfitting on training data?

predictions = model.predict(x_test)
predictions_proba = model.predict_proba(x_test)

# Bagging

bg = BaggingClassifier(SimpleFastTextClassifier(), n_estimators=3)
bg.fit(x_train, y_train)





