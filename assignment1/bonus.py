import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier as DTC

# load the original data into a DataFrame
data = np.loadtxt('original_data.csv', delimiter=' ')
df = pd.DataFrame(data,columns=['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode', 'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp', 'accountcode', 'mail_id', 'ip_id', 'card_id', 'label'])

# compute mean of features for groups of transactions with the
# same creation date timestamp, card id, mail id, and ip id
gdf = df.groupby(['creationdate_stamp']).agg('mean')
gdf2 = df.groupby(['card_id']).agg('mean')
gdf3 = df.groupby(['mail_id']).agg('mean')
gdf4 = df.groupby(['ip_id']).agg('mean')#

# fix DataFrames so that the grouping feature becomes a column again
gdf = gdf.reset_index()
gdf2 = gdf2.reset_index()
gdf3 = gdf3.reset_index()
gdf4 = gdf4.reset_index()


# iterate initial transaction list and append the mean amount
# spent for that particular ip, email, card and time
agg = []
for i in range(df.shape[0]):
    result4 = gdf4.loc[gdf4['ip_id'] == df.loc[i]['ip_id'], 'amount']
    result3 = gdf3.loc[gdf3['mail_id'] == df.loc[i]['mail_id'], 'amount']
    result2 = gdf2.loc[gdf2['card_id'] == df.loc[i]['card_id'], 'amount']
    result = gdf.loc[gdf['creationdate_stamp'] == df.loc[i]['creationdate_stamp'], 'amount']
    agg.append(list(df.loc[i]) + list(result) + list(result2) + list(result3) + list(result4))

# create the augmented dataset DataFrame
df2 = pd.DataFrame(agg, columns=['issuercountry', 'txvariantcode', 'issuer_id', 'amount', 'currencycode', 'shoppercountry', 'interaction', 'verification', 'cvcresponse', 'creationdate_stamp', 'accountcode', 'mail_id', 'ip_id', 'card_id', 'label', 'mean_amount_per_date', 'mean_amount_per_car', 'mean_amount_per_mail', 'mean_amount_per_ip'])

# initialize classification metrics
TP = 0
FP = 0
FN = 0
TN = 0
F1 = 0
j = 0

# save dataset to a new .csv file
df2.to_csv('augemented_data.csv')

# create NumPy arrays for the classification
X = np.array(df2)
# convert continuous label values to binary values
Y = np.array(pd.cut(df2['label'], 2, labels=[0,1]))

# initialize stratified 10-fold cross-validation
cv = StratifiedKFold(n_splits=10)

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
