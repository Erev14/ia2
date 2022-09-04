import numpy as np
import matplotlib.pyplot as plt 
from linear_neuron import Linear_Neuron

# p = 200
# x = -1 + 2 * np.random.rand(p).reshape(1,-1)
# y = -18 * x + 6 + 3 * np.random.rand(p)
# plt.plot(x,y, 'b.')

# neuron = Linear_Neuron(1, 0.1)
# neuron.fit(x, y, solver='BGD')
# xn = np.array([[-1, 1]])
# plt.plot(xn.ravel(), neuron.predict(xn), '--r')
# plt.show()

# neuron = Linear_Neuron(1, 0.1)
# neuron.fit(x, y, solver='SGD')
# xn = np.array([[-1, 1]])
# plt.plot(x,y, 'b.')
# plt.plot(xn.ravel(), neuron.predict(xn), '--r')
# plt.show()

# neuron = Linear_Neuron(1, 0.1)
# neuron.fit(x, y, solver='Pseudo-Inverse')
# xn = np.array([[-1, 1]])
# plt.plot(x,y, 'b.')
# plt.plot(xn.ravel(), neuron.predict(xn), '--r')
# plt.show()


########### For batch
p = 200
x = -1 + 2 * np.random.rand(p).reshape(1,-1)
y = -18 * x + 6 + 3 * np.random.rand(p)
plt.plot(x,y, 'b.')

neuron = Linear_Neuron(1, 0.1)
neuron.fit(x, y, batch_size=10)
xn = np.array([[-1, 1]])
plt.plot(x,y, 'b.')
plt.plot(xn.ravel(), neuron.predict(xn), '--r')
plt.show()