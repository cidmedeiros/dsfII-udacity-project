import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
from time import time
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred','deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive','restricted_stock',
                 'director_fees','to_messages','from_poi_to_this_person','from_messages',
                 'from_this_person_to_poi', 'shared_receipt_with_poi']

data_dict = {}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

data_dict.pop('TOTAL') #outlier removal

data = featureFormat(data_dict, features_list)

y, X = targetFeatureSplit(data)
X_labels = features_list[1:]

mod = LogisticRegression()
rfe = RFE(mod, 13)
fit = rfe.fit(X, y)

print('Number of Selected Features: {}'.format(fit.n_features_), '\n')

supported_features = fit.support_
selected_features = []

for i, feat in enumerate(X_labels):
    if supported_features[i] == True:
        selected_features.append(feat)
        print('{} got selected'.format(feat), '\n')
    else:
        print('{} not selected'.format(feat), '\n')

#test RFE/Logistic Regression with feature scaling
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)
fit_scaled = rfe.fit(X_scaled, y)

print('Number of Scaled Selected Features: {}'.format(fit.n_features_), '\n')

supported_scaled_features = fit_scaled.support_
selected_scaled_features = []

for i, feat in enumerate(X_labels):
    if supported_scaled_features[i] == True:
        selected_scaled_features.append(feat)
        print('{} got selected'.format(feat), '\n')
    else:
        print('{} not selected'.format(feat), '\n')

###selected features

features_list = selected_scaled_features

### Task 2: Remove outliers

#I'm not performing any outliers removal beyond data_dict.pop('TOTAL') because
#I'm actually looking for anomalities in the the data. See report for more information

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#new feature 1: fraction_to_this_person_from_poi
#new feature 2: fraction_from_this_person_poi

for k in data_dict:
    to_messages = data_dict[k]['to_messages']
    from_messages = data_dict[k]['from_messages']
    from_poi = data_dict[k]['from_poi_to_this_person']
    to_poi = data_dict[k]['from_this_person_to_poi']
    if from_poi != 'NaN' and to_messages != 'NaN':
        data_dict[k]['fraction_to_this_person_from_poi'] = float(from_poi)/float(to_messages)
    else:
        data_dict[k]['fraction_to_this_person_from_poi'] = float(0)
        
    if to_poi != 'NaN' and from_messages != 'NaN':
        data_dict[k]['fraction_from_this_person_poi'] = float(to_poi)/float(from_messages)
    else:
        data_dict[k]['fraction_from_this_person_poi'] = float(0)
        

my_dataset = data_dict

features_list.append('fraction_to_this_person_from_poi')
features_list.append('fraction_from_this_person_poi')
features_list.insert(0,'poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

#Split the data into training and testset
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

##SCALING THE DATA
"""
#temporarily removing the features already in scale (fraction_to_this_person_from_poi, fraction_from_this_person_poi)

b = len(features[0]) - 2
p_features_train = [sublist[:b] for sublist in features_train]
p_features_test = [sublist[:b] for sublist in features_test]
"""
#as with all transformation it's important to fit the scaler to the training data only.
scaler.fit(features_train)

#then apply to the data
features_train = scaler.transform(features_train)
features_test = scaler.transform(features_test)

#Frame train set
df_features_train = pd.DataFrame(features_train)

try:
    features_train = df_features_train.to_numpy()
except AttributeError:
    features_train = df_features_train.as_matrix()

#Frame train set
df_features_test = pd.DataFrame(features_test)

try:
    features_test = df_features_test.to_numpy()
except AttributeError:
    features_test = df_features_test.as_matrix()

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#Gaussian Naive Bayes

from sklearn.naive_bayes import GaussianNB
clf_1 = GaussianNB()
clf_1.fit(features_train, labels_train)

pred_1 = clf_1.predict(features_test)
###Evaluating the accuracy
print('Gaussian Naive Bayes: ', accuracy_score(pred_1, labels_test))

##########################################
#Support Vector Machine
from sklearn.svm import SVC
clf = SVC(C= 100.0, kernel='linear')
start2 = time()
clf.fit(features_train, labels_train)
end2 = time()

start3 = time()
pred = clf.predict(features_test)
end3 = time()

time = (end2 - start2)
time1 = (end3 - start3)
print('Time for training: ',time,'s','Time for predicting: ',time1,'s')

###Evaluating the accuracy
print('Support Vector Machine: ', accuracy_score(pred, labels_test))


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)