#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.linear_model import Lasso
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

def outlierCleaner(y_pred_train, x_train, y_train):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (x_train, y_train, error).
    """
    from operator import itemgetter
    
    pair_up = []
    
    for i in range (len(x_train)):
        pair_up.append((x_train[i], y_train[i],
                        ((y_pred_train[i]-y_train[i])**2)))
        
    cleaned_data = sorted(pair_up, key=itemgetter(2), reverse=True)
    c = round(len(cleaned_data) - len(cleaned_data)*0.95)
    cleaned_data = cleaned_data[c:]

    return cleaned_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


###selected features

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred',
                 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                 'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person','from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi']

### Task 2: Remove outliers
    
features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', 'loan_advances',
            'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
            'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
            'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person',
            'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi'] # You will need to use more features

data_dict = {}

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

data = featureFormat(data_dict, features_list, remove_all_zeroes=False)

df = pd.DataFrame(data_dict).T
df.reset_index(inplace=True)
df.rename(columns = {'index': 'person'}, inplace=True)
df.poi = np.where(df.poi == False, float(0), float(1))

frames = []

for col in df.columns:
    if col not in ('poi', 'person', 'email_address'):
        exec('df.{} = np.where(df.{} == "NaN", float(0), df.{})'.format(col, col, col))
        exec('df["{}"] = df["{}"].astype(float)'.format(col, col))

for i, feat in enumerate(features_list):
    if feat != 'poi':
        poi = data[:,0]
        poi = np.reshape(np.array(poi), (len(poi), 1))
        feature = data[:, i]
        feature = np.reshape(np.array(feature), (len(feature), 1))
        reg = LinearRegression()
        reg.fit(feature, poi)
        pred = reg.predict(feature)
        try:
            plt.plot(feature, pred, color="blue")
        except NameError:
            pass
        print('{}: Scatter for Uncleaned Data'.format(feat))
        plt.scatter(feature, poi)
        plt.xlabel(feat)
        plt.ylabel('poi')
        plt.show()
        cleaned_data = outlierCleaner(pred, poi, feature) #remove outliers
        poi_cleaned, feature_cleaned, errors = zip(*cleaned_data)
        reg.fit(feature_cleaned, poi_cleaned)
        pred_cleaned = reg.predict(feature_cleaned)
        try:
            plt.plot(feature_cleaned, pred_cleaned, color="blue")
        except NameError:
            pass
        print('{}: Scatter for Cleaned Data'.format(feat))
        plt.scatter(feature_cleaned, poi_cleaned)
        plt.xlabel(feat)
        plt.ylabel('poi')
        plt.show()
        list_temp = []
        for n, j in zip(poi_cleaned, feature_cleaned):
            list_temp.append([n, j])
            exec('df_{} = pd.DataFrame(list_temp, columns=["poi", "{}"])'.format(feat, feat))
        #DataFrame Wrangling
        exec('df_{}.poi = df_{}.poi.apply(lambda x: float(x))'.format(feat, feat))
        exec('df_{}.{} = df_{}.{}.apply(lambda x: float(x))'.format(feat, feat, feat, feat))
        exec('df_{}["{}_out"] = "selected"'.format(feat, feat))
        exec('df_{} = pd.merge(df_{}, df, on=["poi", "{}"], how="outer")'.format(feat, feat, feat))
        exec('df_{} = df_{}.drop_duplicates()'.format(feat, feat))
        exec('df_{} = df_{}[["{}", "{}_out", "person"]]'.format(feat, feat, feat, feat))
        exec('df_{} = df_{}.set_index("person")'.format(feat, feat))
        exec('frames.append(df_{})'.format(feat))

df_ = df[['person', 'poi']].copy().set_index('person')
semi_df = df_.join(frames)

semi_df['selected'] = np.where(((semi_df.salary_out == 'selected') & (semi_df.deferral_payments_out == 'selected') &
                                (semi_df.total_payments_out == 'selected') & (semi_df.loan_advances_out == 'selected') &
                                (semi_df.bonus_out == 'selected') & (semi_df.restricted_stock_deferred_out == 'selected') &
                                (semi_df.deferred_income_out == 'selected') & (semi_df.total_stock_value_out == 'selected') &
                                (semi_df.expenses_out == 'selected') & (semi_df.exercised_stock_options_out == 'selected') &
                                (semi_df.other_out == 'selected') & (semi_df.long_term_incentive_out == 'selected') &
                                (semi_df.restricted_stock_out == 'selected') & (semi_df.director_fees_out == 'selected') &
                                (semi_df.to_messages_out == 'selected') & (semi_df.from_poi_to_this_person_out == 'selected') &
                                (semi_df.from_messages_out == 'selected') & (semi_df.from_this_person_to_poi_out == 'selected') &
                                (semi_df.shared_receipt_with_poi_out == 'selected')), True, False)

df_final = semi_df.loc[semi_df.selected == True].copy().sort_index()

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