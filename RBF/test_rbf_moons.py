import numpy as np
import matplotlib.pyplot as plt
from rbf import RBFNN
import pandas as pd

def draw(X, Y, net, name):
  plt.figure()
  for i in range(X.shape[1]):
    if Y[0, i] == 0:
      plt.plot(X[0,i], X[1,i], 'ro', markersize=9)
    else:
      plt.plot(X[0,i], X[1,i], 'bo', markersize=9)
  xmin, xmax = np.min(X[0,:])-0.5, np.max(X[0,:])+0.5
  ymin, ymax = np.min(X[1,:])-0.5, np.max(X[1,:])+0.5
  xx, yy = np.meshgrid(np.linspace(xmin, xmax, 200),
                       np.linspace(ymin, ymax, 200))
  data = np.asanyarray([xx.ravel(), yy.ravel()]).T
  zz = net.predict(data)
  zz = zz.reshape(xx.shape)
  plt.contour(xx, yy, zz, [0.5], colors='k', linestyles='--', linewidths=2)
  plt.contour(xx, yy, zz, alpha=0.8, cmap=plt.cm.RdBu)
  plt.xlim([xmin, xmax])
  plt.ylim([ymin, ymax])
  plt.grid()
  plt.savefig(name)

df = pd.read_csv('moons.csv')

X = df.iloc[:, :2].values
Y = df.iloc[:, 2:].values

for i in range(2, 7):
  net = RBFNN(i)
  net.fit(X, Y)

  draw(X.T, Y.T, net, f"rbf_moons_{i}.png")
