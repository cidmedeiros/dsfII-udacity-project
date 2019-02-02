#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import tree
from sklearn.metrics import accuracy_score

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]

#### initial visualization
plt.xlim(0.0, 1.0)
plt.ylim(0.0, 1.0)
plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
plt.legend()
plt.xlabel("bumpiness")
plt.ylabel("grade")
plt.show()
################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

#K nearest neighbors
knn_clf = KNeighborsClassifier(n_neighbors=5)

#knn_clf.fit(features_train, labels_train)

#pred_knn = knn_clf.predict(features_test)

#Random Forest
rnd_clf = RandomForestClassifier()

#rnd_clf.fit(features_train, labels_train)

#pred_rnd = rnd_clf.predict(features_test)

#AdaBoost
ada_clf = AdaBoostClassifier(tree.DecisionTreeClassifier(max_depth=1),
                             n_estimators=50, learning_rate=0.5, algorithm='SAMME.R')

#ada_clf.fit(features_train, labels_train)

#pred_ada = ada_clf.predict(features_test)

#SVC
clf_svc = SVC(C= 1000000.0, kernel='rbf')

#Accuracy and Visu for all
for clf in (knn_clf, rnd_clf, ada_clf, clf_svc):
    clf.fit(features_train, labels_train)
    y_pred = clf.predict(features_test)
    print(clf.__class__.__name__, accuracy_score(y_pred, labels_test))
    
try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
