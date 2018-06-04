import pandas as pd
from saxpy.sax import *
import numpy as np
from matplotlib import pyplot

# load training and testing data
train2 = pd.read_csv('BATADAL_dataset04.csv')
train1 = pd.read_csv('BATADAL_dataset03.csv')

# remove non-sensor columns
sensors = list(train2.columns)
sensors.remove('DATETIME')
sensors.remove('ATT_FLAG')

# store known attacks
labels = list(train2.loc[train2['ATT_FLAG']==1].index)

# remove non-sensor data
train1.index = train1['DATETIME']
series = train1.loc[:,train1.columns!='DATETIME']
train2.index = train2['DATETIME']
series2 = train2.loc[:,train2.columns!='DATETIME']

# store results for each sensor
dsax = {}
for sensor in sensors:
    dsax[sensor] = {}

# iterate over various window, aggregates and alphabet sizes
for win in range(10,11):
    for paa in range(2,3):
        for alpha in range(2,3):
            for sensor in sensors: 
                # discretize training data
                sax1 = sax_via_window(np.array(series[sensor]), win, paa, alpha, "none", 0.01)
                # compute n-gram probabilities
                ngr = list(sax1.keys())
                acc = 0
                nprobs = {}
                for i in ngr:
                    nprobs[i] = len(sax1[i])
                    acc = acc + nprobs[i]
                for i in ngr:
                    nprobs[i] /= acc

                # discretize test data
                sax2 = sax_via_window(np.array(series2[sensor]),win,paa,alpha,"none",0.01)
                # compute n-gram probabilities
                ngr2 = list(sax2.keys())
                acc2=0
                nprobs2 = {}
                for i in ngr2:
                    nprobs2[i] = len(sax2[i])
                    acc2 = acc2 + nprobs2[i]
                for i in ngr2:
                    nprobs2[i] /= acc2

                # store possible anomalies
                ano = set()
                anoma = {}
                for i in ngr2:
                    if i not in ngr:
                        for j in sax2[i]:
                            ano.add(j)
                
                # compute performance
                TP = 0
                FP = 0
                for i in list(ano):
                    if i in labels:
                        TP +=1 
                    else:
                        FP += 1
                # store the results of that run
                dsax[sensor][(str(win),str(paa),str(alpha))] = (str(TP),str(FP))

print(dsax)
