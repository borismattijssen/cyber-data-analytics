#!/usr/bin/python3 

import numpy as np
from sklearn.metrics import roc_curve
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC as SVC
import matplotlib.pyplot as plt

# read .csv file into an array
data = np.loadtxt('original_data.csv',delimiter=' ')

# simple aggregation based on IP, just adds the total number of transactions of that IP as a feature
agg = {}
for item in data:
    if item[12] in agg.keys():
        agg[item[12]] = agg[item[12]] + 1
    else:
        agg[item[12]] = 1

agg_data = []
for i in range(data.shape[0]):
    agg_data.append(np.append(data[i][:-1],agg[data[i][12]]))

# separate features from labels for the original and aggregated data sets
x2 = data[:,:data.shape[1]-2]
x1 = np.array(agg_data)
y = data[:,data.shape[1]-1]

# turn them into arrays
X = np.asarray(x2,dtype=np.int32)
X1 = np.asarray(x1,dtype=np.int32)
Y = np.asarray(y,dtype=np.int32)

# use a random forest classifier with 10-fold cross-validation for both datasets
clf = RFC()
scores = cross_val_score(clf, X1,Y , cv=10)
print(scores)

clf = RFC()
scores = cross_val_score(clf, X, Y, cv=10)
print(scores)

# perform SMOTE and then repeat the above process
# it's wrong, needs to separate training and test before using SMOTE
sm = SMOTE()
X_res, Y_res = sm.fit_sample(X,Y)
clf = RFC()
scores = cross_val_score(clf, X_res, Y_res, cv=10)
print(scores)

sm = SMOTE()
X_res, Y_res = sm.fit_sample(X1,Y)
clf = RFC()
scores = cross_val_score(clf, X_res, Y_res, cv=10)
print(scores)

# separate training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# train stochastic gradient descent classifier
clf = SGD()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

# calculate roc curve
fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

# create plot
plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocSGD.png')

# same process for gaussian naive bayes classifier
clf = GNB()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocGNB.png')

# same for decision tree classifier
clf = DTC()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocDTC.png')
# repeat for SMOTED data
X_train, X_test, Y_train, Y_test = train_test_split(X_res,Y_res,test_size=0.2)

clf = SGD()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocSGDs.png')

clf = GNB()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocGNBs.png')

clf = DTC()
clf.fit(X_train,Y_train)
Y_labels = clf.predict(X_test)

fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)

plt.figure()
plt.plot(fpr,tpr,label='ROC curve')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.savefig('RocDTCs.png')
