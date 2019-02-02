#!/usr/bin/python 

""" 
    Skeleton code for k-means clustering mini-project.
"""

import pickle
import numpy
import matplotlib.pyplot as plt
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from featureScaling import featureScaling
import pandas as pd

def unix_dos_pikle(address):
    original = address
    destination = address
    content = ''
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
        with open(destination, 'wb') as output:
            for line in content.splitlines():
                outsize += len(line) + 1
                output.write(line + str.encode('\n'))

def Draw(pred, features, poi, mark_poi=False, name="image.png", f1_name="feature 1", f2_name="feature 2"):
    """ some plotting code designed to help you visualize your clusters """

    ### plot each cluster with a different color--add more colors for
    ### drawing more than five clusters
    colors = ["b", "c", "k", "m", "g"]
    for ii in list(range(pred.n_clusters)):
        plt.scatter(features[ii][0], features[ii][1], color = colors[ii])

    ### if you like, place red stars over points that are POIs (just for funsies)
    if mark_poi:
        for ii in list(range(pred.n_clusters)):
            if poi[ii]:
                plt.scatter(features[ii][0], features[ii][1], color="r", marker="*")
    plt.xlabel(f1_name)
    plt.ylabel(f2_name)
    plt.savefig(name)
    plt.show()



### load in the dict of dicts containing all the data on each person in the dataset
unix_dos_pikle("../final_project/final_project_dataset.pkl")
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "rb") )
### there's an outlier--remove it! 
data_dict.pop("TOTAL", 0)


### the input features we want to use 
### can be any key in the person-level dictionary (salary, director_fees, etc.) 
feature_1 = "salary"
feature_2 = "exercised_stock_options"
#feature_3 = "total_payments"
poi  = "poi"
features_list = [poi, feature_1, feature_2]
data = featureFormat(data_dict, features_list )

#Rescaling the features
#data[:, 1] = numpy.array(featureScaling(data[:, 1])).reshape(len(data[:,1]))
#data[:, 2] = numpy.array(featureScaling(data[:, 2])).reshape(len(data[:,2]))

arr_1 = data[:, 1]

arr_2 = data[:, 2]


#splitting data
poi, finance_features = targetFeatureSplit(data)


### in the "clustering with 3 features" part of the mini-project,
### you'll want to change this line to 
### for f1, f2, _ in finance_features:
### (as it's currently written, the line below assumes 2 features)
for f1, f2 in finance_features:
    plt.scatter(f1, f2)
plt.show()

### cluster here; create predictions of the cluster labels
### for the data and store them to a list called pred
from sklearn.cluster import KMeans
pred = KMeans(n_clusters = 2)
pred.fit(data)

### rename the "name" parameter when you change the number of features
### so that the figure gets saved to a different file
try:
    Draw(pred, finance_features, poi, mark_poi=False, name="clusters.pdf", f1_name=feature_1, f2_name=feature_2)
except NameError:
    print ("no predictions object named pred found, no clusters to plot")
    
#my_dict_2 = {i: numpy.where(pred.labels_ == i)[0] for i in range(pred.n_clusters)}
my_dict_3 = {i: numpy.where(pred.labels_ == i)[0] for i in range(pred.n_clusters)}

df = pd.DataFrame(data_dict).T
