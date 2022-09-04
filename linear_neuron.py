import numpy as np

class Linear_Neuron():
    def __init__(self, n_inputs, learning_rate=0.1) -> None:
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.etha = learning_rate

    def predict(self, X):
        return np.dot(self.w, X) + self.b # Y_est

    def batcher(self, X, Y, size):
        p = X.shape[1]
        li, ui = 0, size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + size, ui + size
            else:
                return None
    
    def MSE(self, X, Y):
        p = X.shape[1]
        Y_est = self.predict(X)
        return (1/p) * np.sum((Y-Y_est)**2)

    def fit(self, X, Y, epochs: int = 50, batch_size = 24):
        mse_history = []
        for _ in range(epochs):
            minibatch = self.batcher(X, Y, batch_size)
            for mX, mY in minibatch:
                p = mX.shape[1]
                Y_est = self.predict(mX)
                self.w += (self.etha / p) * np.dot((mY - Y_est), mX.T).ravel()
                self.w += (self.etha / p) * np.sum((mY - Y_est))
            mse_history.append(self.MSE(X, Y))

        return mse_history

        
            
