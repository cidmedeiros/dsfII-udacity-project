import sys
import pickle
sys.path.append("../tools/")
from time import time
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import make_scorer, recall_score, precision_score, f1_score

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

#function do display the selected features
def display_selected(labels, supported_features):
    selected_features = []
    for i, feat in enumerate(labels):
        if supported_features[i] == True:
            selected_features.append(feat)
            print('{} got selected'.format(feat), '\n')
        else:
            print('{} not selected'.format(feat), '\n')
    return selected_features

#RFE/Logistic Regression
def rfe_selection (X, y, n_features):
    mod = LogisticRegression()
    rfe = RFE(mod, n_features)
    fit = rfe.fit(X, y)
    supported_features = fit.support_
    return supported_features

#Runninf RFE selection without scaling
supported_features = rfe_selection(X, y, n_features=10) #boolean list
selected_features = display_selected(X_labels, supported_features) #list with the selected features names

#RFE/Logistic Regression with feature scaling
def feat_scaler(X, t=X):
    scaler = StandardScaler()
    scaler.fit(X)
    if X != t:
        X_scaled = scaler.transform(t)
    if X == t:
        X_scaled = scaler.transform(X)        
    return X_scaled

#Runninf RFE selection with scaling
X_scaled = feat_scaler(X)
supported_scaled_features = rfe_selection(X_scaled, y, n_features=13) #boolean list

selected_scaled_features = display_selected(X_labels, supported_scaled_features) #list with the selected features names

###selected features
features_list = selected_scaled_features

### Task 2: Remove outliers

#I'm not performing any outliers removal beyond data_dict.pop('TOTAL') because
#I'm actually looking for anomalities in the the data. See report for more information

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

#Create new feature 1: fraction_to_this_person_from_poi
#Create new feature 2: fraction_from_this_person_poi
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
        

my_dataset = data_dict #new features added, TOTAL outlier removed, data not yet definitive scaled

#finalizes the features_list
features_list.append('fraction_to_this_person_from_poi')
features_list.append('fraction_from_this_person_poi')
features_list.insert(0,'poi')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)

#Split the data into training and testset (data only with the selected features; data not scaled yet)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

##SCALING THE DATA
#as with all transformation it's important to fit the scaler to the training data only
#then apply to the testing data

features_test = feat_scaler(features_train, t=features_test)

#scaling-transforming the training data
features_train = feat_scaler(features_train, t=features_train)

#Running PCA
#set whether to run the pca or not
run_pca = False

if run_pca == True:
    from sklearn.decomposition import PCA
    pca = PCA(n_components='mle', svd_solver='full')
    pca.fit(features_train)
    features_train = pca.transform(features_train)
    features_test = pca.transform(features_test)
    
def display(scores, name):
    print('')
    print(name, '\n')
    print('Scores:', scores)
    print('Score Mean:', scores.mean())
    print('Score STD:', scores.std())

#makeing scorers from the metrics
precision = make_scorer(precision_score)
recall = make_scorer(recall_score)
f1 = make_scorer(f1_score)

##########################################
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB # Provided to give you a starting point. Try a variety of classifiers.

clf_1 = GaussianNB()
acc = cross_val_score(clf_1, features_train, labels_train, scoring = recall, cv=10)
    
display(acc, 'Naive Bayes')

##########################################
#Support Vector Machine
from sklearn.svm import SVC

clf_2 = SVC(kernel='rbf', gamma = 0.001, C= 50)
acc = cross_val_score(clf_2, features_train, labels_train, scoring = recall, cv=10)

display(acc, 'SVM-SVC')
    
##########################################
#Decision Trees

#cross validation
from sklearn import tree

clf_3 = tree.DecisionTreeClassifier(min_samples_split=40)
acc = cross_val_score(clf_3, features_train, labels_train, scoring = recall, cv=10)
    
display(acc, 'Decision Trees')

##########################################
#K Near Neighbors
from sklearn.neighbors import KNeighborsClassifier

clf_4 = KNeighborsClassifier(n_neighbors=3)
acc = cross_val_score(clf_4, features_train, labels_train, scoring = recall, cv=10)

display(acc, 'K Near Neighbors')

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

###RandomizedSearchCV
# Utility function to report best scores extracted from sklearn documentation
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
            
n_iter_search = 20 
##########################################
#Support Vector Machine
print('RandomSearch for SVM-SVC')
clf_SVC = SVC()

param_svc = {'C': [1.0, 10, 50, 100, 200, 500, 800, 1000], 
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
             'gamma': [0.001, 0.010, 0.1, 1]}

random_search_svc = RandomizedSearchCV(clf_SVC, param_distributions=param_svc, n_iter=n_iter_search,
                                   scoring = recall, cv=10)

random_search_svc.fit(features_train, labels_train)

report(random_search_svc.cv_results_, n_top=3)

##########################################
#Decision Trees
print('RandomSearch for Decision Tree')
clf_tree = tree.DecisionTreeClassifier()

param_tree = {'min_samples_split':[2, 4, 8], 'min_samples_leaf':[2, 4, 8],
              'max_features':[3, 5, 7, 9]}

random_search_tree = RandomizedSearchCV(clf_tree, param_distributions=param_tree,n_iter=n_iter_search,
                                   scoring = recall, cv=10)

random_search_tree.fit(features_train, labels_train)

report(random_search_tree.cv_results_, n_top=3)

##########################################
#K-nearest Neighbor
n_iter_search = 18 #KNN maximum splits for this dataset
print('RandomSearch for K-nearest Neighbor')
clf_knn = KNeighborsClassifier()

param_knn = {'n_neighbors':[3,5,9],'weights':['uniform', 'distance'],'leaf_size':[10, 15, 30]}

random_search_knn = RandomizedSearchCV(clf_knn, param_distributions=param_knn,n_iter=n_iter_search,
                                       scoring = recall, cv=10)

random_search_knn.fit(features_train, labels_train)

report(random_search_knn.cv_results_, n_top=3)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.
clf = tree.DecisionTreeClassifier(min_samples_split=8, min_samples_leaf=8, max_features=9)

clf.fit(features_train, labels_train)
y = clf.predict(features_test)
importances = clf.feature_importances_

print('Final Precision Score On the Test Set: ', precision_score(y, labels_test), '\n')
print('Final Recall Score On the Test Set: ', recall_score(y, labels_test), '\n')
for i, v in enumerate(features_list[1:]):
    print('Feature {} holds importance of '.format(v), importances[i])

dump_classifier_and_data(clf, my_dataset, features_list)