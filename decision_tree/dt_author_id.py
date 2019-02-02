#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from new_line_dos_unix import new_line_dos_unix
from email_preprocess import preprocess
from sklearn import tree
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
new_line_dos_unix()
features_train, features_test, labels_train, labels_test = preprocess()

#########################################################
### your code goes here ###
start = time()
clf = tree.DecisionTreeClassifier(min_samples_split=40)

clf.fit(features_train, labels_train)
end = time()

start1 = time()
pred = clf.predict(features_test)
end1=time()

acc = accuracy_score(pred, labels_test)

print(acc)

time = (end - start)
time1 = (end1 - start1)
print('Time for training: ',time,'s','Time for predicting: ',time1,'s')

#########################################################


