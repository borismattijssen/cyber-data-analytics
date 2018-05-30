import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARMA
from sklearn.metrics import mean_squared_error

df = pd.read_csv('training_1.csv')
df = df.drop(columns=['ATT_FLAG', 'DATETIME'])

p_from = 1
p_to = 5

q_from = 1
q_to = 5

best = {}
for signal in list(df):
    best_aic = sys.float_info.max
    best_pq = ""
    for p in range(p_from,p_to):
        for q in range(q_from,q_to):
            try:
                model = ARMA(df[signal], order=(p,q))
                model_fit = model.fit(disp=-1)
                print("{} ({},{}): {}".format(signal, p, q, model_fit.aic))
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_pq = "{},{}".format(p,q)
            except:
                print("Got exception for signal {}, with ({},{})".format(signal, p, q))
    print("Best AIC for {}: {} ({})".format(signal, best_aic, best_pq))
    best[signal] = {
        'aic': best_aic,
        'pq': best_pq
    }

with open('/output/best.json', 'w') as f:
    json.dump(best, f)
