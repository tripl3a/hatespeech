#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 10:31:47 2018

@author: arndt
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingClassifier
import sklearn.metrics as skm
from os import chdir
chdir("/home/arndt/git-reps/hatespeech/")

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

#%%

# Scikit Bagging

bg = BaggingClassifier(skift.FirstObjFtClassifier(), n_estimators=3)
bg.fit(X_train, y_train) #...doesn't work...

#%%

# Build multiple models using K-Folds

import numpy as np
from sklearn import model_selection

seed = 77
kfold = model_selection.KFold(n_splits=10, random_state=seed)
#kfold = model_selection.KFold(n_splits=3, shuffle=True)

# build multiple models using k folds:
kfold_clfs = list()
for train_index, test_index in kfold.split(df_train):
    X = pd.DataFrame(df_train.loc[:,"comment_text"])
    y = df_train.loc[:,"toxic"]
    clf = skift.FirstObjFtClassifier(lr=0.2)
    clf.fit(X.iloc[train_index], y.iloc[train_index])
    print(clf.score(X.iloc[test_index], y.iloc[test_index]))
    kfold_clfs.append(clf)

#%%

# Make predictions with a list of classifiers on dataframe X
def ensemble_predict_proba(classifiers, X):
    proba = [classifier.predict_proba(X) for classifier in classifiers]
    mean = np.zeros(proba[0].shape)
    for i in range(len(classifiers)):
        mean = mean + proba[i]
    mean = mean / float(len(classifiers))
    return mean

kfold_proba = ensemble_predict_proba(kfold_clfs, X_test)
kfold_labels = np.zeros(kfold_proba.shape[0]) #initialize array
kfold_labels[kfold_proba[:,0]<=kfold_proba[:,1]] = 1
score_preds(y_test, kfold_labels)

#%%

# Implement bagging manually
# https://machinelearningmastery.com/implement-bagging-scratch-python/


# Bootstrap Aggregation Algorithm
def bagging(X_train, y_train, X_test, frac_sample_size, n_classifiers, bootstrap=True):
    classifiers = list()
    for i in range(n_classifiers):
        X_sample = X_train.sample(frac=frac_sample_size, replace=bootstrap)
        y_sample = X_sample.join(y_train)["toxic"] #inner join on index
        clf = skift.FirstObjFtClassifier(lr=0.2)
        clf.fit(X_sample, y_sample)
        classifiers.append(clf)
    predictions = ensemble_predict_proba(classifiers, X_test)
    return classifiers, predictions

def bagging_scores(classifiers, X, y):
    scores = [classifier.score(X, y) for classifier in classifiers]
    return scores

#%%

clfs, bg_proba = bagging(X_train, y_train, X_test, 1, 3)
bg_scores_train = bagging_scores(clfs, X_train, y_train)
bg_scores_test = bagging_scores(clfs, X_test, y_test)

bg_proba[bg_proba[:,0] > bg_proba[:,1]].shape
bg_proba[bg_proba[:,0] <= bg_proba[:,1]].shape

bg_preds = np.zeros(bg_proba.shape[0])

test_str = "this is a fucking test"
ensemble_predict_proba(clfs, pd.DataFrame([test_str]))
test_str = "i love you"
ensemble_predict_proba(clfs, pd.DataFrame([test_str]))
test_str = "eat shit, asshole!!!"
ensemble_predict_proba(clfs, pd.DataFrame([test_str]))

#%%

from sklearn.ensemble import VotingClassifier
# Voting Classifier - Multiple Model Ensemble

etms = list()
for i, item in enumerate(clfs):
    etms.append(("ft"+str(i),item))

evc = VotingClassifier(estimators=etms, voting='soft') #recommended for an ensemble of well-calibrated classifiers
evc.fit(X_train.iloc[1:4000], y_train.iloc[1:4000])
evc.score(X_test, y_test)

evc.predict(X_test)
evc.predict_proba(X_test)

