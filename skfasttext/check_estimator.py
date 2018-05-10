#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  4 11:43:24 2018

@author: arndt
"""

from os import chdir
chdir("/home/arndt/git-reps/hatespeech/")
from skfasttext import SimpleFastTextClassifier

# You can check whether your estimator adheres to the scikit-learn interface 
# and standards by running utils.estimator_checks.check_estimator on the class:
from sklearn.utils.estimator_checks import check_estimator
check_estimator(SimpleFastTextClassifier)  # passes?
