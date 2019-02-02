#!/usr/bin/python

import pickle
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat
from operator import itemgetter

### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb"))
data_dict.pop('TOTAL', 0)

df_outliers = (pd.DataFrame(data_dict)).T
df_outliers = df_outliers.query('salary != "NaN"').copy()

features_list = ['salary', 'deferral_payments', 'total_payments', 'loan_advances',
            'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value',
            'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
            'restricted_stock', 'director_fees','to_messages','from_poi_to_this_person',
            'from_messages', 'from_this_person_to_poi','shared_receipt_with_poi']

list_poi = {}

for feat in features_list:
    features = ['poi']
    features.append(feat)
    data = featureFormat(data_dict, features)
    for point in data:
        x = point[0]
        y = point[1]
        matplotlib.pyplot.scatter(x, y)
        
    matplotlib.pyplot.xlabel('{}'.format(features[0]))
    matplotlib.pyplot.ylabel('{}'.format(features[1]))
    matplotlib.pyplot.show()
    
    poi = []
    val= []
    no_val = []
    for point in data:
        if point[0] == 1:
            poi.append(point[0])
            val.append(point[1])
        else:
            no_val.append(point[1])
    
    poi = np.array(poi)
    poi[poi == 0] = np.nan
    val = np.array(val)
    val[val == 0] = np.nan
    no_val = np.array(no_val)
    no_val[no_val == 0] = np.nan
        
    list_poi[feat] = sum(poi), (np.nanmean(val), np.nanmean(no_val))