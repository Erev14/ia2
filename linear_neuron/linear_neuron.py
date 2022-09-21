import numpy as np

class LinearNeuron():
    def __init__(self, n_inputs, learning_rate=0.1) -> None:
        MIN_VAL = -1
        RANDOM_RANGE = 2 # start from range + min and will be the max value
        self.w = MIN_VAL + RANDOM_RANGE * np.random.rand(n_inputs)
        self.b = MIN_VAL + RANDOM_RANGE * np.random.rand()
        self.etha = learning_rate
        self.solvers = {
            'SGD': self.sgd,
            'BGD': self.bgd
        }
    
    def batcher(self, X, Y, size):
        FIRST_ELEMENT = 0
        p = X.shape[1]
        li, ui = FIRST_ELEMENT, size
        while True:
            if li < p:
                yield X[:, li:ui], Y[:, li:ui]
                li, ui = li + size, ui + size
            else:
                return None
    
    def mse(self, X, Y):
        p = X.shape[1]
        Y_est = self.predict(X)
        return (1/p) * np.sum((Y-Y_est)**2)

    def predict(self, X):
        return self.w @ X + self.b # Y_est

    def sgd(self, X, Y, p):
        for i in range(p): 
            y_est = self.predict(X[:,i]) # predict for single data
            self.w += self.etha * (Y[:,i] - y_est) * X[:,i]
            self.b += self.etha * (Y[:,i] - y_est)
    
    def bgd(self, X, Y, p):
        Y_est = self.predict(X)
        self.w += (self.etha / p) * ((Y - Y_est) @ X.T).ravel() # /p for normalize
        self.b += (self.etha / p) * np.sum(Y - Y_est) # /p for normalize

    def pseudoinverse(self, X, Y, p): #(direct method)
        X_hat = np.concatenate((np.ones((1,p)), X), axis=0) # tupple and on axies of rows
        W_hat = Y.reshape(1, -1) @ np.linalg.pinv(X_hat)
        self.b = W_hat[0,0]
        self.w = W_hat[0,1:]
    
    def fit(self, X, Y, epochs: int = 50, batch_size = 24, solver='SGD'):
        MIN_BATCH = 1
        if batch_size < MIN_BATCH:
            batch_size = X.shape[1] # full set (no batch)
        
        mse_history = []
        if solver == 'pseudoinverse' or solver not in self.solvers:
            self.pseudoinverse(X, Y, X.shape[1])
            mse_history.append(self.mse(X, Y))
            return mse_history
        
        for _ in range(epochs):
            minibatch = self.batcher(X, Y, batch_size)
            for mX, mY in minibatch:
                p = mX.shape[1]
                self.solvers[solver](mX, mY, p)
            mse_history.append(self.mse(X, Y))
        return mse_history