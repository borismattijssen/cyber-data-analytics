#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import roc_curve
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC

def draw_roc(fig, X_train, X_test, Y_train, Y_test, title):
    # train classifier
    clf.fit(X_train,Y_train)
    Y_labels = clf.predict(X_test)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)


    plt.plot(fpr,tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

def draw_rocs(clf, clf_name, X, Y):
    # separate training and test set
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

    # perform SMOTE
    sm = SMOTE()
    X_res, Y_res = sm.fit_sample(X_train,Y_train)

    fig = plt.figure()
    # create plot
    draw_roc(fig, X_train, X_test, Y_train, Y_test, '%s UNSMOTEd' % clf_name)
    draw_roc(fig, X_res, X_test, Y_res, Y_test, '%s SMOTEd' % clf_name)

    fig.savefig('ROC_%s.png' % clf_name)




# read .csv file into an array
data = np.loadtxt('data/original_data.csv', delimiter=' ')

# separate features from labels for the original and aggregated data sets
x2 = data[:,:data.shape[1]-2]
y = data[:,data.shape[1]-1]

# turn them into arrays
X = np.asarray(x2,dtype=np.int32)
Y = np.asarray(y,dtype=np.int32)

draw_rocs(SGD(), 'SGD', X, Y)




# use a random forest classifier with 10-fold cross-validation for both datasets
# clf = RFC()
# scores = cross_val_score(clf, X, Y, cv=10)
# print(scores)

# perform SMOTE and then repeat the above process
# it's wrong, needs to separate training and test before using SMOTE

#
# clf = RFC()
#
# # train stochastic gradient descent classifier
# clf = SGD()
#
# # same process for gaussian naive bayes classifier
# clf = GNB()
# clf.fit(X_train,Y_train)
# Y_labels = clf.predict(X_test)
#
# fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)
#
# plt.figure()
# plt.plot(fpr,tpr,label='ROC curve')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.savefig('RocGNB.png')
#
# # same for decision tree classifier
# clf = DTC()
# clf.fit(X_train,Y_train)
# Y_labels = clf.predict(X_test)
#
# fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)
#
# plt.figure()
# plt.plot(fpr,tpr,label='ROC curve')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.savefig('RocDTC.png')
# # repeat for SMOTED data
# X_train, X_test, Y_train, Y_test = train_test_split(X_res,Y_res,test_size=0.2)
#
# clf = SGD()
# clf.fit(X_train,Y_train)
# Y_labels = clf.predict(X_test)
#
# fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)
#
# plt.figure()
# plt.plot(fpr,tpr,label='ROC curve')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.savefig('RocSGDs.png')
#
# clf = GNB()
# clf.fit(X_train,Y_train)
# Y_labels = clf.predict(X_test)
#
# fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)
#
# plt.figure()
# plt.plot(fpr,tpr,label='ROC curve')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.savefig('RocGNBs.png')
#
# clf = DTC()
# clf.fit(X_train,Y_train)
# Y_labels = clf.predict(X_test)
#
# fpr, tpr, thresholds = roc_curve(Y_test,Y_labels)
#
# plt.figure()
# plt.plot(fpr,tpr,label='ROC curve')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver operating characteristic example')
# plt.savefig('RocDTCs.png')
