#!/usr/bin/python3 

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import BaggingClassifier


# read .csv file into an array
data = np.loadtxt('data/original_data.csv',delimiter=' ')

# separate features from labels for the original and aggregated data sets
x = data[:,:data.shape[1]-2]
y = data[:,data.shape[1]-1]

# turn them into arrays
X = np.asarray(x,dtype=np.int32)
Y = np.asarray(y,dtype=np.int32)

# initialize simple classification metrics
TP = 0
FP = 0
FN = 0
TN = 0
F1 = 0
j = 0

# threshold where positive class probabilities start to be considered indicative of positive class
cutoff = 0.1

# prepare for 10-fold stratified cross-validation
cv = StratifiedKFold(n_splits=10)

# split each fold into a training and test set
for train, test in cv.split(X,Y):
    j += 1

    # create and train bagging decision tree classifier
    clf = BaggingClassifier(DTC(), max_samples=0.61, max_features=0.61)
    clf.fit(X[train],Y[train])
    # predict classifier labels
    Y_labels = (clf.predict_proba(X[test])[:,1] > cutoff).astype(int)

    # compute and output simple metrics
    for i in range(len(Y_labels)):
        if Y[test][i] == 1 and Y_labels[i] == 1:
            TP += 1
        if Y[test][i] == 0 and Y_labels[i] == 1:
            FP += 1
        if Y[test][i] == 1 and Y_labels[i] == 0:
            FN += 1
        if Y[test][i] == 0 and Y_labels[i] == 0:
            TN += 1
    F1 += f1_score(Y[test],Y_labels)
    print('Iteration: ' + str(j))
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))
    print('F1: ' + str(F1/j))