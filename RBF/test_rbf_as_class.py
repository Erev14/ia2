from tkinter import ON
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from rbf import RBFNN
from sklearn.datasets import make_moons

x, y = make_moons(400, noise=0.2)
y_e = OneHotEncoder().fit_transform(y[:, None]).toarray()

net = RBFNN(250)
net.fit(x, y_e)

plt.figure()
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdBu)
xmin, xmax = np.min(x[:, 0]) - 0.5, np.max(x[:, 0]) + 0.5
ymin, ymax = np.min(x[:, 1]) - 0.5, np.max(x[:, 1]) + 0.5

xx, yy = np.meshgrid(np.linspace(xmin, xmax, 500),
                     np.linspace(ymin, ymax, 500))

data = np.concatenate((xx.ravel().reshape(-1, 1),
                       yy.ravel().reshape(-1, 1)), axis=1)

zz = net.predict_class(data)
zz = zz.reshape(xx.shape)
plt.contourf(xx, yy, zz, alpha=0.7, cmap=plt.cm.RdBu)
plt.xlim([xmin, xmax])
plt.ylim([ymin, ymax])
plt.show()