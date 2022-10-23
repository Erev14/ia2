import numpy as np
from MLP import MLP

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])

Y = np.array([[0, 1, 1, 1]])

net = MLP((2, 50, 1), output_activation = 'logistic')
print(net.predict(X))

net.fit(X, Y)

print(net.predict(X))
