#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:28:35 2018

@author: alexei, arndt
"""

from os import chdir
chdir("/home/arndt/git-reps/hatespeech/")

import sklearn.metrics as skm
import fastText
import pandas as pd
import csv

df=pd.read_csv("data/kaggle-data.csv")

# data preperation
df=df[["id","comment_text","toxic"]]
df["label"]="__label__not_toxic"
df.loc[df["toxic"]==1,"label"]="__label__toxic"
df["comment_text"]=df["comment_text"].apply(str.replace,args=("\n"," "))
df["comment_text"]=df["comment_text"].apply(str.replace,args=("\"",""))

def train_model(train_index, test_index):
    df[["label","comment_text"]].iloc[train_index].to_csv("data/train_data.txt",index=False,sep=" ",header=False,escapechar=" ",quoting=csv.QUOTE_NONE)
    df[["comment_text"]].iloc[test_index].to_csv("data/test_data.txt",index=False,sep=" ",header=False,escapechar=" ",quoting=csv.QUOTE_NONE)
    return fastText.train_supervised("data/train_data.txt")
    
def score_model(model, test_index):
    test_labels=df["label"].iloc[test_index].apply(str.replace,args=("__label__",""))
    pred_labels=df["comment_text"].iloc[test_index].apply(lambda x:model.predict(x)[0][0]).apply(str.replace,args=("__label__",""))
    print("confusion matrix:")
    print(str(skm.confusion_matrix(test_labels, pred_labels)))
    print(str(skm.classification_report(test_labels, pred_labels)))
    print("f1 macro: %0.4f" % (skm.precision_recall_fscore_support(test_labels, pred_labels, average='macro')[2]))
    print("f1 micro: %0.4f" % (skm.precision_recall_fscore_support(test_labels, pred_labels, average='micro')[2]))
    print("\n")

def get_predictions(model, test_index, k=1):
    df[["label","comment_text"]].iloc[test_index].to_csv("test_data.txt",index=False,sep=" ",header=False,escapechar=" ",quoting=csv.QUOTE_NONE)
    return df["comment_text"].iloc[test_index].apply(lambda x:model.predict(x, k=k))

#%%

# Build multiple models using K-Folds

import numpy as np
from sklearn import model_selection

seed = 77
kfold = model_selection.KFold(n_splits=3, random_state=seed)
#kfold = model_selection.KFold(n_splits=3, shuffle=True)

for train_index, test_index in kfold.split(df):
    model = train_model(train_index, test_index)
    score_model(model, test_index)



