import numpy as np
import matplotlib.pyplot as plt
from linear_neuron import LinearNeuron

POINTS = 200
NOISE = 3
x = -1 + 2 * np.random.rand(POINTS).reshape(1,-1)
y = -18 * x + 6 + NOISE * np.random.rand(POINTS)
plt.plot(x, y, 'b.')

neuron = LinearNeuron(1, 0.1)
history = neuron.fit(x, y, solver='BGD', batch_size=0)
xn = np.array([[-1, 1]])
plt.plot(xn.ravel(), neuron.predict(xn), '--r')

plt.figure()
plt.plot(history)
plt.title('Batch')

plt.show()