import numpy as np
import matplotlib.pyplot as plt
from rbf import RBFNN


p = 200
xl = -5
xu = 5

x = np.linspace(xl, xu, p).reshape(-1, 1)
y = 2 * np.cos(x) + np.sin(3*x) + 5

net = RBFNN(2)
net.fit(X, Y)


p = 1000
xnew = np.linspace(xl, xu, p).reshape(-1, 1)
ynew = net.predict(xnew)

plt.plot(xnew, ynew, '-b')
plt.plot(x, y, 'or')
plt.show()