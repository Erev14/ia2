import numpy as np
from active_functions import *

class MLP:

    def __init__(self, layers_dims: tuple, 
            hidden_activation = 'tanh', 
            output_activation = 'logistic') -> None:
        # Attibutes
        self.L = len(layers_dims) - 1
        self.w = [None] * (self.L + 1)
        self.b = [None] * (self.L + 1)
        self.f = [None] * (self.L + 1)

        for l in range(1, self.L + 1):
            self.w[l] = -1 + 2 * np.random.rand(layers_dims[l], 
                                                layers_dims[l-1])
            self.b[l] = -1 + 2 * np.random.rand(layers_dims[l], 1)
            if l == self.L:
                self.f[l] = functions[output_activation]
            else:
                self.f[l] = functions[hidden_activation]
    def predict(self, X):
        A = np.asanyarray(X)
        for l in range(1, self.L + 1):
            Z = self.w[l] @ A + self.b[l]
            A = self.f[l](Z)
        return A
    
    def fit(self, X, Y, epochs=500, learning_rate=0.1):
        p = X.shape[1]

        for _ in range(epochs):
            A = [None] * (self.L + 1)
            dA = [None] * (self.L + 1)
            lg = [None] * (self.L + 1)

            # propagation
            A[0] = X.copy()

            for l in range(1, self.L + 1):
                Z = self.w[l] @ A[l-1] + self.b[l]
                A[l], dA[l] = self.f[l](Z, derivative=True)
            
            # back propagation
            for l in range(self.L, 0, -1):
                if l == self.L:
                    lg[l] = (Y - A[l]) * dA[l]
                else:
                    lg[l] = (self.w[l+1].T @ lg[l+1]) * dA[l]

            # desent gradient
            for l in range(1, self.L + 1):
                self.w[l] += (learning_rate / p) * (lg[l] @ A[l-1].T)
                self.b[l] += (learning_rate / p) * np.sum(lg[l])