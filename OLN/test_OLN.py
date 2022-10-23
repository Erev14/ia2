import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from OLN import OLN
import matplotlib as plt

df = pd.read_csv('EDGAR JOEL AREVALO CHAVARIN - Dataset_A03.csv')

X = df.iloc[:, 0:2].values
Y = df.iloc[:, 2:6].values

print(df.head())
print(X.shape)
print(Y.shape)

neuron = OLN(X.shape[1], Y.shape[1])
neuron.train(X, Y)