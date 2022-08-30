import numpy as np

class Linear_Neuron():
    def __init__(self, n_inputs, learning_rate=0.1) -> None:
        self.w = -1 + 2 * np.random.rand(n_inputs)
        self.b = -1 + 2 * np.random.rand()
        self.etha = learning_rate

    def predict(self, X):
        return np.dot(self.w, X) + self.b # Y_est
    
    def fit(self, X, Y, epochs: int = 50, solver='BGD'):
        p = X.shape[1] # columns (patron)

        if solver == 'SGD':
            for _ in range(epochs):
                for i in range(p): 
                    y_est = self.predict(X[:,i]) # predict for single data
                    self.w += self.etha * (Y[:,i] - y_est) * X[:,i]
                    self.b += self.etha * (Y[:,i] - y_est)
        elif solver == 'BGD':
            for _ in range(epochs):
                Y_est = self.predict(X)
                self.w += (self.etha / p) * ((Y - Y_est) @ X.T).ravel() # /p for normalize
                self.b += (self.etha / p) * np.sum(Y - Y_est) # /p for normalize
        else: # Pseudo-Inverse (direct method)
            X_hat = np.concatenate((np.ones((1,p)), X), axis=0) # tupple and on axies of rows
            W_hat = np.dot(Y.reshape(1, -1), np.linalg.pinv(X_hat))
            self.b = W_hat[0,0]
            self.w = W_hat[0,1:]
            
