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
import fastText as ft

chdir(path.dirname(path.abspath( '__file__ ')))
df_test = pd.merge(pd.read_csv("data/test_unidecode.csv"),
                   pd.read_csv("data/test_labels.csv"),
                   how="inner",
                   on="id")

df_train=pd.read_csv("data/train_unidecode.csv")
df_fix=pd.read_csv("data/train.csv")
class_name = "toxic"
#class_name = "identity_hate"

# data preperation
#df_train=df_train[["id","comment_text","toxic"]]
#df_train["label"]="__label__not_toxic"
#df_train.loc[df_train["toxic"]==1,"label"]="__label__toxic"
df_train["comment_text"]=df_train["comment_text"].apply(str.replace,args=("\n"," "))
df_train["comment_text"]=df_train["comment_text"].apply(str.replace,args=("\"",""))

df_test["comment_text"]=df_test["comment_text"].apply(str.replace,args=("\n"," "))
df_test["comment_text"]=df_test["comment_text"].apply(str.replace,args=("\"",""))

X_train = pd.DataFrame(df_train.loc[:,"comment_text"])
y_train = df_fix.loc[:,class_name]
X_test = pd.DataFrame(df_test[df_test[class_name]>-1].loc[:,"comment_text"])
y_test = df_test[df_test[class_name]>-1].loc[:,class_name]

#%%

# Single skift model

import skift
#import skfasttext as skft

def print_results(N, p, r):
    print("Number of documents\t" + str(N))
    print("Precision@{}\t{:.3f}".format(1, p))
    print("Recall@{}\t{:.3f}".format(1, r))

'''
wiki_model = ft.train_supervised(input="data/train_data.txt", pretrainedVectors="data/wiki.en/wiki.en.vec data/wiki-news-300d-1M.vec", dim=300)

# Prints results with test and training data
print(" \n\n Results with Pre-trained vectors from wikipedia \n ")
print_results(*wiki_model.test("data/test.txt"))
print("\n Results with the trained data \n ")
print_results(*wiki_model.test("data/train.txt"))
'''

skift_clf = skift.FirstObjFtClassifier(wordNgrams=2, maxn=3, pretrainedVectors="data/wiki-news-300d-1M-subword.vec,"
                                                         "data/wiki.en/wiki.en.vec,data/crawl-300d-2M.vec", dim=300)
skift_clf.fit(X_train,y_train)
#skift_clf = skift.FirstObjFtClassifier(minn=3, maxn=3)
#skift_clf.fit(X_train, y_train)

print("score on test data: %0.4f" % (skift_clf.score(X_test, y_test)))
print("score on training data: %0.4f" % (skift_clf.score(X_train, y_train))) #model overfitting on training data?

#print("score on test data w/ pre: %0.4f" % (skift_pre.score(X_test, y_test)))
#print("score on training data w/ pre: %0.4f" % (skift_pre.score(X_train, y_train))) #model overfitting on training data?


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
#score_preds(y_test, np.zeros(y_test.shape)) #majority class classifier
