import pandas as pd
from OLN import OLN
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('EDGAR JOEL AREVALO CHAVARIN - Dataset_A03.csv')

X = df.iloc[:, 0:2].values.T
Y = df.iloc[:, 2:6].values.T

neuron = OLN(X.shape[0], Y.shape[0], 'logistic')
neuron.train(X, Y, epochs=300, learning_rate=.7)

Ypred = neuron.predict(X)

colors =[[1,0,1], [0,1,0], [0,0,1], [0,0,0]]

plt.figure()
ax1 = plt.subplot(1, 2, 1)
y_c = np.argmax(Y, axis=0)
for i in range(X.shape[1]):
	ax1.plot(X[0,i], X[1,i], 'o', c=colors[y_c[i]], markersize=9)
ax1.axis([-5.5,5.5,-5.5,5.5])
ax1.set_title('Problema original')
ax1.grid()

ax2 = plt.subplot(1,2,2)
y_c = np.argmax(Ypred, axis=0)
for i in range(X.shape[1]):
	ax2.plot(X[0,i], X[1,i], 'o', c=colors[y_c[i]])
ax2.axis([-5.5,5.5,-5.5,5.5])
ax2.set_title('Prediccion de la red')
ax2.grid()

plt.show()