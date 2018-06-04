import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer
import numpy as np
from matplotlib import pyplot as plt

# load train and test data
train2 = pd.read_csv('BATADAL_dataset04.csv')
train1 = pd.read_csv('BATADAL_dataset03.csv')

# remove non-sensor column names
sensors = list(train2.columns)
sensors.remove('DATETIME')
sensors.remove('ATT_FLAG')

# store known attacks
labels = list(train2.loc[train2['ATT_FLAG']==1].index)

# remove non-sensor data
train1.index = train1['DATETIME']
train1 = train1.loc[:,train1.columns!='ATT_FLAG']
series = train1.loc[:,train1.columns!='DATETIME']
train2.index = train2['DATETIME']
train2 = train2.loc[:,train2.columns!='ATT_FLAG']
series2 = train2.loc[:,train2.columns!='DATETIME']

# normalize training set 1 for pca
nm = Normalizer()
nseries =pd.DataFrame(nm.fit_transform(series))

# normalize training set 2 for pca
nm2 = Normalizer()
nseries2 = pd.DataFrame(nm2.fit_transform(series2))

# list of sensors to exclude
excl = list(series.columns)

# initialization of residuals for all sensors
resids = {}
cseries3 = series
cseries4 = series2

# loop until we have enough sensors excluded
while len(excl) > 15:
    for sensor in excl:
        # perform pca and get residuals for all sensors
        cseries = cseries3.drop(labels=sensor,axis=1)
        cseries2 = cseries4.drop(labels=sensor,axis=1)

        nm3 = Normalizer()
        nseries3 =pd.DataFrame(nm3.fit_transform(cseries))

        nm4 = Normalizer()
        nseries4 =pd.DataFrame(nm3.fit_transform(cseries2))

        mnm = 100000
        # find the sensor with the maximum residual for various principal components
        for i in range(1,15):
            pca2 = PCA(n_components=i)
            pca2.fit(nseries3)
            projection = pca2.transform(nseries3)

            components = pca2.components_
            recreation = np.matmul(projection,components)
            residuals = np.subtract(np.array(nseries3),recreation)

            residual = np.sum(np.square(residuals),axis=1)

            if np.max(residual) < mnm:
                mnm = np.max(residual)
        
        resids[sensor] = mnm
    
    # remove the sensor with the maximum residuals sum
    delete = ''
    for sensor in list(resids.keys()):
        if resids[sensor] < mnm:
            mnm = resids[sensor]
            delete = sensor
    
    excl.remove(delete)
    print(delete)
    cseries3 = cseries3.drop(labels=delete,axis=1)
    cseries4 = cseries4.drop(labels=delete,axis=1)
    resids = {}

    #repform pca, this time to get performance
    nm3 = Normalizer()
    nseries3 =pd.DataFrame(nm3.fit_transform(cseries3))

    nm4 = Normalizer()
    nseries4 =pd.DataFrame(nm3.fit_transform(cseries4))


    for i in range(1,15):
        print(i)
        pca2 = PCA(n_components=i)
        pca2.fit(nseries3)
        projection = pca2.transform(nseries3)
        projection2 = pca2.transform(nseries4)

        components = pca2.components_
        recreation2 =  np.matmul(projection2,components)
        recreation = np.matmul(projection,components)
        residuals2 = np.subtract(np.array(nseries4),recreation2)
        residuals = np.subtract(np.array(nseries3),recreation)

        residual2 = np.sum(np.square(residuals2),axis=1)
        residual = np.sum(np.square(residuals),axis=1)
        temp =  np.cumsum(np.square(residuals),axis=1)

        ind = np.where(np.square(residual2)>np.max(np.square(residual)))
        rows = ind[0]
        unique_rows = set(rows)
        unique_rows = list(unique_rows)
        unique_rows.sort()

        TP = 0
        FP = 0
        possible_anomalies = []
        for j in unique_rows:
            if j in labels:
                TP +=1
                possible_anomalies.append(j)
            else:
                FP +=1

        ind = np.where(np.square(residual2)<np.min(np.square(residual)))
        rows = ind[0]
        unique_rows = set(rows)
        unique_rows = list(unique_rows)
        unique_rows.sort()

        for j in unique_rows:
            if j in labels:
                TP +=1
                possible_anomalies.append(j)
            else:
                FP +=1
        print('TP '  +str(TP))
        print('FP ' + str(FP))
        
