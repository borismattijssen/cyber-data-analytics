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

def draw_roc(clf, X_train, X_test, Y_train, Y_test, title):
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

def draw_rocs(clf, clf_name, X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test):
    fig = plt.figure()
    # create plot
    plt.subplot(1,2,1)
    draw_roc(clf, X_train, X_test, Y_train, Y_test, '%s UNSMOTEd' % clf_name)
    plt.subplot(1,2,2)
    draw_roc(clf, X_smote_train, X_test, Y_smote_train, Y_test, '%s SMOTEd' % clf_name)

    fig.savefig('ROC_%s.png' % clf_name)




# read .csv file into an array
data = np.loadtxt('data/original_data.csv', delimiter=' ')

# separate features from labels for the original and aggregated data sets
x2 = data[:,:data.shape[1]-2]
y = data[:,data.shape[1]-1]

# turn them into arrays
X = np.asarray(x2,dtype=np.int32)
Y = np.asarray(y,dtype=np.int32)

# separate training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# perform SMOTE
sm = SMOTE()
X_smote_train, Y_smote_train = sm.fit_sample(X_train,Y_train)

# draw_rocs(SGD(), 'SGD', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
draw_rocs(RFC(), 'RFC', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
draw_rocs(GNB(), 'GNB', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
draw_rocs(DTC(), 'DTC', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
