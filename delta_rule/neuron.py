from active_functions import functions
import numpy as np

class neuron:
    def __init__(self, 
            n_inputs, 
            act_function='linear'
        ) -> None:
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.f = act_function

    def predict_probability(self, X, threshold=0.5):
        if self.f != 'logistic':
            return None
        return 1 * (self.predict_train(X) > threshold)
        
    def predict(self, X):
        Z = self.w @ X + self.b
        return functions[self.f](Z)

    def train(self, 
            X, 
            Y, 
            epochs=100,
            L2_penality=0, 
            learning_rate=0.1
        ):
        p = X.shape[1] # columns
        for _ in range(epochs):
            # Propagation
            Z = self.w @ X + self.b
            Yest, dY = self.f(Z, derivative=True)

            # training
            lg = (Y - Yest) * dY # haddlemand product

            self.w = (1 - L2_penality * learning_rate) \
                     * self.w + (learning_rate / p) \
                     * (lg @ X.T).ravel()
            self.b =+ (learning_rate / p) * np.sum(lg)