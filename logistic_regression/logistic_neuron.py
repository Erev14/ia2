import numpy as np

class LogisticNeuron():
  def __init__(self, n_inputs, learning_rate=0.1) -> None:
    MIN_VAL = -1
    RANDOM_RANGE = 2 # start from range + min and will be the max value
    self.w = MIN_VAL + RANDOM_RANGE * np.random.rand(n_inputs)
    self.b = MIN_VAL + RANDOM_RANGE * np.random.rand()
    self.etha = learning_rate

  def predict_train(self, X):
    Z = self.w @ X + self.b
    return 1 / (1 + np.exp(-Z)) # y_est

  def predict(self, X, threshold=0.5):
    return 1 * (self.predict_train(X) > threshold)
  
  def fit(self, X, Y, epochs: int = 50):
    p = X.shape[1]
    for _ in range(epochs):
      Y_est = self.predict(X)
      self.w += (self.etha / p) * ((Y - Y_est) @ X.T).ravel() # /p for normalize
      self.b += (self.etha / p) * np.sum(Y - Y_est) # /p for normalize