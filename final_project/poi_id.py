import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression


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

data_dict.pop('TOTAL')

data = featureFormat(data_dict, features_list)

X = data[:, 1:]
X_labels = features_list[1:]
y = data[:, 0]

#Plot distribution with its respective linear discriminant coeficient

clf_sf = LinearDiscriminantAnalysis(n_components=19)

clf_sf.fit(X, y)

coef_list = clf_sf.coef_

for label, coef in zip(X_labels, coef_list):
    print(label)
    plt.hist(X[:, X_labels.index(label)])
    plt.show()
    print('{} = {}'.format(label, coef), '\n')

###selected features

### Task 2: Remove outliers

df = pd.DataFrame(data_dict).T
df.reset_index(inplace=True)
df.rename(columns = {'index': 'person'}, inplace=True)
df.poi = np.where(df.poi == False, float(0), float(1))


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

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