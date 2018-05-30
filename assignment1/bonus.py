import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

# load the augmented data
df = pd.read_csv('data/augmented_data.csv')

# create NumPy arrays for the classification
X = np.array(df.loc[:,df.columns!='label'])
# convert continuous label values to binary values
Y = np.array(pd.cut(df['label'], 2, labels=[0,1]))

# initialize stratified 10-fold cross-validation
cv = StratifiedKFold(n_splits=10)

# initialize classification metrics
TP = 0
FP = 0
FN = 0
TN = 0
F1 = 0
j = 0

# repeat classification for each fold
for train, test in cv.split(X,Y):
    j+=1
    # create and train bagging decision tree classifier
    clf = BaggingClassifier(DTC(), max_samples=0.61, max_features=0.61)
    clf.fit(X[train],Y[train])

    # validate on test set
    Y_labels = clf.predict(X[test])

    # compute classification metrics
    for i in range(len(Y_labels)):
        if Y[test][i] == 1 and Y_labels[i] == 1:
            TP += 1
        if Y[test][i] == 0 and Y_labels[i] == 1:
            FP += 1
        if Y[test][i] == 1 and Y_labels[i] == 0:
            FN += 1
        if Y[test][i] == 0 and Y_labels[i] == 0:
            TN += 1

    F1 += f1_score(Y[test], Y_labels)
    print('Iteration: ' + str(j))
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))
    print('F1: ' + str(F1/j))

# load original preprocessed data
data = np.loadtxt('data/encoded_data.csv',delimiter=' ')

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

# initialize stratified 10-fold cross-validation
cv = StratifiedKFold(n_splits=10)

# probabilities threshold
cutoff = 0.1
# repeat classification for each fold
for train, test in cv.split(X,Y):
    j+=1
    # create and train bagging decision tree classifier
    clf = BaggingClassifier(DTC(), max_samples=0.61, max_features=0.61)

    clf.fit(X[train],Y[train])

    # post-process probabilities
    Y_labels = (clf.predict_proba(X[test])[:,1] > cutoff).astype(int)

    # compute classification metrics
    for i in range(len(Y_labels)):
        if Y[test][i] == 1 and Y_labels[i] == 1:
            TP += 1
        if Y[test][i] == 0 and Y_labels[i] == 1:
            FP += 1
        if Y[test][i] == 1 and Y_labels[i] == 0:
            FN += 1
        if Y[test][i] == 0 and Y_labels[i] == 0:
            TN += 1

    F1 += f1_score(Y[test], Y_labels)
    print('Iteration: ' + str(j))
    print('TP: ' + str(TP))
    print('FP: ' + str(FP))
    print('FN: ' + str(FN))
    print('TN: ' + str(TN))
    print('F1: ' + str(F1/j))
