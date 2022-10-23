import numpy as np

def linear(z, derivative=False):
    a = z # activate
    if derivative:
        da = np.ones(z.shape)
        return a, da
    return a

def logistic(z, derivative=False):
    a = 1 / ( 1 + np.exp(-z) ) # activate
    if derivative:
        da = a * (1 - a)
        return a, da
    return a

def tanh(z, derivative=False):
    a = np.tanh(z) # activate
    if derivative:
        da = (1 + a) * (1 - a)
        return a, da
    return a

def relu(z, derivative=False):
    a = z * (z >= 0) # activate
    if derivative:
        da = np.array(z >= 0, dtype=float)
        return a, da
    return a

def softmax(z, derivative=False):
    e = np.exp(z - np.max(z, axis=0))
    a = e/np.sum(e)
    if derivative:
        da = np.ones(z.shape, dtype=float)
        return a, da
    return a

functions = {
    'linear': linear,
    'logistic': logistic,
    'tanh': tanh,
    'relu': relu,
    'softmax': softmax
}