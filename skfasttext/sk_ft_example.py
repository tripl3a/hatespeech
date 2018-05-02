#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  2 17:34:43 2018

@author: arndt
"""


# http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator
# https://github.com/vishnumani2009/sklearn-fasttext

from os import chdir
chdir("/home/arndt/git-reps/nlp/01_hatespeech/")
from skfasttext import SimpleFastTextClassifier

# files were previously created in the HateSpeech.py script
train_file="data/train_data.txt"
test_file="data/test_data.txt"

clf=SimpleFastTextClassifier()
model = clf.fit(train_file)
predictions = clf.predict(test_file, k_best=2)