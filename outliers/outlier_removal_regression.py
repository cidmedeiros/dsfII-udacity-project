#!/usr/bin/python

import numpy
import matplotlib.pyplot as plt
import pickle

from outlier_cleaner import outlierCleaner

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
 
### load up some practice data with outliers in it
#unix_dos_pikle("practice_outliers_ages.pkl")
#unix_dos_pikle("practice_outliers_net_worths.pkl")
ages = pickle.load( open("practice_outliers_ages.pkl", "rb") )
net_worths = pickle.load( open("practice_outliers_net_worths.pkl", "rb") )

### ages and net_worths need to be reshaped into 2D numpy arrays
### second argument of reshape command is a tuple of integers: (n_rows, n_columns)
### by convention, n_rows is the number of data points
### and n_columns is the number of features
ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))
from sklearn.cross_validation import train_test_split
ages_train, ages_test, net_worths_train, net_worths_test = train_test_split(ages, net_worths, test_size=0.1, random_state=42)

### fill in a regression here!  Name the regression object reg so that
### the plotting code below works, and you can see what your regression looks like
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(ages_train,net_worths_train)

try:
    plt.plot(ages, reg.predict(ages), color="blue")
    print('Scatter for Uncleaned Data')
except NameError:
    pass
plt.scatter(ages, net_worths)
plt.xlabel('Age')
plt.ylabel('Networth')
plt.show()

print('R-squared: ', reg.score(ages_test,net_worths_test))

print('Slope: ', reg.coef_)

print('Intercept: ', reg.intercept_)
print('')

### identify and remove the most outlier-y points
cleaned_data = []
try:
    predictions = reg.predict(ages_train)
    cleaned_data = outlierCleaner(predictions, ages_train, net_worths_train)
except NameError:
    print ("your regression object doesn't exist, or isn't name reg")
    print ("can't make predictions to use in identifying outliers")


### only run this code if cleaned_data is returning data
if len(cleaned_data) > 0:
    ages, net_worths, errors = zip(*cleaned_data)
    ages = numpy.reshape(numpy.array(ages), (len(ages), 1))
    net_worths = numpy.reshape(numpy.array(net_worths), (len(net_worths), 1))

    ### refit your cleaned data!
    try:
        reg.fit(ages, net_worths)
        plt.plot(ages, reg.predict(ages), color="blue")
        print('Scatter for the Cleaned Data')
    except NameError:
        print ("you don't seem to have regression imported/created,")
        print ("   or else your regression object isn't named reg")
        print ("   either way, only draw the scatter plot of the cleaned data")
    plt.scatter(ages, net_worths)
    plt.xlabel("ages")
    plt.ylabel("net worths")
    plt.show()

else:
    print ("outlierCleaner() is returning an empty list, no refitting to be done")


print('R-squared: ', reg.score(ages_test,net_worths_test))

print('Slope: ', reg.coef_)

print('Intercept: ', reg.intercept_)
