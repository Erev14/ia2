import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from logistic_neuron import LogisticNeuron
from sklearn.metrics import accuracy_score

df = pd.read_csv('EDGAR JOEL AREVALO CHAVARIN - diabetes.csv')

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)

neuron = LogisticNeuron(X_train.shape[1])
neuron.fit(X_train.T, y_train.T, epochs=500)

print("Train score: ", accuracy_score(y_train.T, neuron.predict(X_train.T)))

print("Test score: ", accuracy_score(y_test.T, neuron.predict(X_test.T)))