from logistic_neuron import LogisticNeuron
import numpy as np

X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 1]])
neuron = LogisticNeuron(2, 1)
print(neuron.predict(X))

neuron.fit(X, Y, epochs=500)
print(neuron.predict(X))