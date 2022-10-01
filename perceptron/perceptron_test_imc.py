from perceptron import Perceptron
import numpy as np
import matplotlib.pyplot as plt

def draw_2d_neuron(neuron):
    w1, w2, b = neuron.w[0], neuron.w[1], neuron.b
    plt.plot([-2, 2], [ (-1/w2) * (-w1*2+b), (-1/w2) * (w1*2+b) ], '--k')

def train_and_plot(X, Y, testing: str):
    neuron = Perceptron(2, 0.1)
    print(neuron.predict(X))

    neuron.fit(X, Y)
    print(neuron.predict(X))

    plt.figure()

    p = X.shape[1]
    for i in range(p):
        if Y[i] == 1:
            plt.plot(X[0, i], X[1,i], 'or')
        else:
            plt.plot(X[0, i], X[1,i], 'ob')

    plt.title('El perceptron ' + testing)
    plt.grid('on')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')

    draw_2d_neuron(neuron)
    plt.show()
    return neuron

def scale(X, x_min, x_max):
    return np.array([
        (X[0, :]  - x_min[0]) / (x_max[0] - x_min[0]),
        (X[1, :]  - x_min[1]) / (x_max[1] - x_min[1]),
    ])

num_samples = 125
max_weigh = 120
max_height = 2.4
min_weigh = 40
min_height = 1.2
X = np.array([
        min_weigh + (max_weigh - min_weigh) * np.random.rand(num_samples), # weigh
        min_height + (max_height - min_height) * np.random.rand(num_samples) # height
    ]).round(1)

IMC = np.array(X[0]/X[1]**2).round(2)
Y = np.where(IMC > 25, 1, 0)

x_max = X.max(axis=1)
x_min = X.min(axis=1)

X_scale = scale(X, x_min, x_max)

neuron = train_and_plot(X_scale, Y, "IMC")

new_data = np.array([
    np.array([83, 54]),
    np.array([1.59, 1.63]),
])
X_scale = scale(new_data, x_min, x_max)
print("new predict")
print(neuron.predict(new_data))