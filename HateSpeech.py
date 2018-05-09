#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 16:28:35 2018

@author: alexei, arndt
"""

import numpy as np
import pandas as pd
import sklearn.metrics as skm
from os import chdir, path

chdir(path.dirname(path.abspath( __file__ )))

df_test = pd.merge(pd.read_csv("data/test.csv"),
                   pd.read_csv("data/test_labels.csv"),
                   how="inner",
                   on="id")

df_train=pd.read_csv("data/train.csv")

# data preperation
df_train=df_train[["id","comment_text","toxic"]]
df_train["label"]="__label__not_toxic"
df_train.loc[df_train["toxic"]==1,"label"]="__label__toxic"
df_train["comment_text"]=df_train["comment_text"].apply(str.replace,args=("\n"," "))
df_train["comment_text"]=df_train["comment_text"].apply(str.replace,args=("\"",""))

df_test["comment_text"]=df_test["comment_text"].apply(str.replace,args=("\n"," "))
df_test["comment_text"]=df_test["comment_text"].apply(str.replace,args=("\"",""))

X_train = pd.DataFrame(df_train.loc[:,"comment_text"])
y_train = df_train.loc[:,"toxic"]
X_test = pd.DataFrame(df_test[df_test["toxic"]>-1].loc[:,"comment_text"])
y_test = df_test[df_test["toxic"]>-1].loc[:,"toxic"]

#%%

# Single skift model

import skift

skift_clf = skift.FirstObjFtClassifier(lr=0.2)
skift_clf.fit(X_train, y_train)

print("score on test data: %0.4f" % (skift_clf.score(X_test, y_test)))
print("score on training data: %0.4f" % (skift_clf.score(X_train, y_train))) #model overfitting on training data?

preds = skift_clf.predict(X_test)
preds_proba = skift_clf.predict_proba(X_test)

def score_preds(y_true, y_pred):
    print("confusion matrix:")
    print(str(skm.confusion_matrix(y_true, y_pred)))
    print("classification report:")
    print(str(skm.classification_report(y_true, y_pred)))
    print("f1 macro: %0.4f" % (skm.precision_recall_fscore_support(y_true, y_pred, average='macro')[2]))
    print("f1 micro: %0.4f" % (skm.precision_recall_fscore_support(y_true, y_pred, average='micro')[2]))

score_preds(y_test, preds)

#%%
score_preds(y_test, np.zeros(y_test.shape)) #majority class classifier
