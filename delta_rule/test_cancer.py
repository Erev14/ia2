import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from neuron import Neuron

df = pd.read_csv('EDGAR JOEL AREVALO CHAVARIN - cancer.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values
print(df.head())
print(X.shape)
print(Y.shape)

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
print("X train: ", X_train.shape)

neuron = Neuron(X_train.shape[1], 'logistic')
neuron.train(X_train.T, y_train, epochs=500)

correct = 0
for x, y in zip(X_train, y_train):
  expected = neuron.predict(x)
  if expected == y:
    correct += 1

print("Train score: ", correct / X_train.shape[0])

correct = 0
for x, y in zip(X_test, y_test):
  expected = neuron.predict(x)
  if expected == y:
    correct += 1

print("Test score: ", correct / X_test.shape[0])