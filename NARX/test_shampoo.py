import numpy as np
import matplotlib.pyplot as plt
from NARX import *
import pandas as pd
import statsmodels.api as sm

CSV_NAME = 'sales-of-shampoo-over-a-three-ye.csv'
COL_MAIN_VALUE = 'Sales of shampoo over a three year period'

dataset = pd.read_csv(CSV_NAME)
dataset[COL_MAIN_VALUE] = pd.to_numeric(dataset[COL_MAIN_VALUE],errors='coerce')
temperatures = pd.to_numeric(dataset.values[:,1], errors='ignore')
temperatures = np.array(temperatures).reshape(-1,1)
dates = dataset.values[:,0]
u = temperatures

#Test NARX a un paso
narx = NARX(1, 1, 10,
            dense_hidden_layers=(10, 100),
            learning_rate=0.01, n_repeat_train=5)
# narx = NARX(1, 1, 10,
#             dense_hidden_layers=(10, 100, 50, 100,),
#             learning_rate=0.01, n_repeat_train=1)
y_est = np.zeros((1, dataset.shape[0]))
u = u.T
for i in range(dataset.shape[0]-1):
    x = u[:,i].reshape(-1,1)
    y = u[:,i+1].reshape(-1,1)
    y_est[:,i] = narx.predict_and_train(x, y).ravel()
    
#Grafica
dataset['predict'] = y_est.T

fig, ax = plt.subplots(facecolor='whitesmoke')
dataset[[COL_MAIN_VALUE, 'predict']].plot(figsize=(100, 30), fontsize=12)
legend = plt.legend()
legend.prop.set_size(20)

plt.savefig("shampoo_sales.png", facecolor=fig.get_facecolor(), transparent=True)