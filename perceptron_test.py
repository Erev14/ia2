from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [0, 0, 1, 1],
    [0, 1, 0, 1]
])

Y = np.array([0, 1, 1, 1])

def draw_2d_neuron(neuron):
    w1, w2, b = neuron.w[0], neuron.w[1], neuron.b
    plt.plot([-2, 2], [ (-1 / w2)*(-w1*2+b), (-1/w2)*(w1*2+b) ], '--k')

neuron = Perceptron(2, 0.1)
print(neuron.predict(X))

neuron.fit(X, Y)
# print(neuron.predict(X))

# plt.figure()

# p = X.shape[1]
# for i in range(p):
#     if Y[i] == 1:
#         plt.plot(X[0, i], X[1,i], 'or')
#     else:
#         plt.plot(X[0, i], X[1,i], 'ob')

# plt.title('El perceptron')
# plt.grid('on')
# plt.xlim([-2, 2])
# plt.ylim([-2, 2])
# plt.xlabel(r'$x_1$')
# plt.ylabel(r'$x_2$')

# draw_2d_neuron(neuron)