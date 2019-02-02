#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
import numpy as np
import sys
sys.path.append("../tools/")
from time import time
from email_preprocess import preprocess
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from new_line_dos_unix import new_line_dos_unix

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
new_line_dos_unix()
features_train, features_test, labels_train, labels_test = preprocess()

#features_train = features_train[:round(len(features_train)/100)] 
#labels_train = labels_train[:round(len(labels_train)/100)] 

#features_train = np.array(features_train)
#labels_train = np.array(labels_train)

#features_train.astype(int)
#labels_train.astype(int)

#########################################################
### your code goes here ###
start = time()
clf = SVC(C= 10000.0, kernel='rbf')
clf.fit(features_train, labels_train)
end = time()

start1 = time()
pred = clf.predict(features_test)
end1=time()

accuracy = accuracy_score(pred, labels_test)

print(accuracy)

time = (end - start)
time1 = (end1 - start1)
print('Time for training: ',time,'s','Time for predicting: ',time1,'s')
#########################################################

print(pred[10], pred[26], pred[50])

chris = pred.sum()
