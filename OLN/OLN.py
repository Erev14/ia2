from active_functions import functions
import numpy as np

class OLN:
    def __init__(self, n_inputs, n_outputs, act_function='linear') -> None:
        MIN_VAL = -1
        RANDOM_RANGE = 2 # start from range + min and will be the max value
        self.w = MIN_VAL + RANDOM_RANGE * np.random.rand(n_outputs, n_inputs)
        self.b = MIN_VAL + RANDOM_RANGE * np.random.rand(n_outputs, 1)
        self.f = act_function
        
    def predict(self, X):
        Z = self.w @ X + self.b
        return functions[self.f](Z)

    def train(self, X, Y, epochs=100, learning_rate=0.1):
        p = X.shape[1] # columns
        for _ in range(epochs):
            # Propagation
            Z = self.w @ X + self.b
            Yest, dY = functions[self.f](Z, derivative=True)

            # training
            lg = (Y - Yest) * dY # haddlemand product

            self.w += (learning_rate / p) * lg @ X.T
            self.b += (learning_rate / p) * np.sum(lg, axis=1).reshape(-1, 1)