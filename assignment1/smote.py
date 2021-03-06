#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.neighbors import KNeighborsClassifier

def draw_roc(clf, X_train, X_test, Y_train, Y_test, color, label):
    # train classifier
    clf.fit(X_train, Y_train)
    Y_labels = clf.predict(X_test)

    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_labels)
    # compute area under roc curve
    area =  auc(fpr, tpr)
    # add roc curve in the figure with a different color and a legend label
    plt.plot(fpr, tpr, color=color, label=(label + ' (area = {0:0.2f})').format(area))

def draw_rocs(clf, clf_name, X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test):
    fig = plt.figure()
    # create plots
    draw_roc(clf, X_train, X_test, Y_train, Y_test, 'darkorange','%s UNSMOTEd' % clf_name)
    draw_roc(clf, X_smote_train, X_test, Y_smote_train, Y_test, 'navy','%s SMOTEd' % clf_name)

    #set axis range, title and legend and dashed line of random guessing classifier
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Comparison of SMOTE with ROC curves')
    plt.legend(loc="lower right")

    plt.show()
    fig.savefig('ROC_%s.png' % clf_name)




# read .csv file into an array
data = np.loadtxt('data/original_data.csv', delimiter=' ')

# separate features from labels for the original and aggregated data sets
x = data[:,:data.shape[1]-2]
y = data[:,data.shape[1]-1]

# turn them into arrays
X = np.asarray(x,dtype=np.int32)
Y = np.asarray(y,dtype=np.int32)

# separate training and test set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)

# perform SMOTE
sm = SMOTE()
X_smote_train, Y_smote_train = sm.fit_sample(X_train,Y_train)

# draw ROC curves for Decision Tree Classifier, Gaussian Naive Bayes Classifier and K Nearest Neighbors Classifier
draw_rocs(DTC(), 'DTC', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
draw_rocs(GNB(), 'GNB', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
draw_rocs(KNeighborsClassifier(), '5-NearestNeighbors', X_train, X_smote_train, X_test, Y_train, Y_smote_train, Y_test)
