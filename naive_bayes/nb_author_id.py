#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
import numpy as np
import sys
from timeit import default_timer as timer
sys.path.append("../tools/")
from new_line_dos_unix import new_line_dos_unix
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
new_line_dos_unix()

features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
clf = GaussianNB()
start = timer()
clf.fit(features_train, labels_train)
end = timer()

start1 = timer()
pred = clf.predict(features_test)
end1 = timer()
###Evaluating the accuracy

print(accuracy_score(pred, labels_test))

time = (end - start)
time1 = (end1 - start1)
print('Time for training: ',time,'s','Time for predicting: ',time1,'s')

#########################################################


