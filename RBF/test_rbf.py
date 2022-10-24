import numpy as np
import matplotlib.pyplot as plt
from rbf import RBFNN

p = 200
xl = -5
xu = 5

X = np.linspace(xl, xu, p).reshape(-1, 1)
Y = 2 * np.cos(X) + np.sin(3*X) + 5

net = RBFNN(2)
net.fit(X, Y)

p = 1000
xnew = np.linspace(xl, xu, p).reshape(-1, 1)
ynew = net.predict(xnew)

plt.plot(xnew, ynew, '-b')
plt.plot(X, Y, 'or')
plt.savefig("rbf_xor")