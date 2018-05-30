import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../data/training_1.csv')
plt.matshow(df.corr())
plt.colorbar()
plt.show()
