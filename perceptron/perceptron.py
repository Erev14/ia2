# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 2022

@author: Edgar Joel Arévalo Chavarin (Erev14)
"""

# artificial neuron
import numpy as np

class Perceptron:
    def __init__(self, n_in: int, learning_rate: float):
        self.w = -1 + 2 * np.random.rand(n_in)
        self.b = -1 + 2 * np.random.rand()
        self.etha = learning_rate

    def predict(self, X):
        p = X.shape[1] # columns
        y_est = np.zeros(p)
        for i in range(p):
            y_est[i] = self.w @ X[:, i] + self.b # matrix mult @ or np.dot
            y_est[i] = int(y_est[i] >= 0)
        return y_est

    def fit(self, X, Y, epoch:int = 50):
        p = X.shape[1] # columns
        for _ in range(epoch):
            for i in range(p):
                y_est = self.predict(X[:,i].reshape(-1,1)) # force np to use matrix instead of array
                self.w += self.etha * (Y[i] - y_est) * X[:,i]
                self.b += self.etha * (Y[i] - y_est)

