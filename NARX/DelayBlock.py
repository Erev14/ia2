import numpy as np

class DelayBlock:
    def __init__(self, n_inputs, n_delays):
        self.memory = np.zeros((n_inputs, n_delays))
        
    def add(self, x):
        # Create deep copy to avoid conficts
        # With original sign
        x_new = x.copy().reshape(1,-1)
        
        # add delay
        self.memory[:,1:] = self.memory[:,:-1]
        self.memory[:,0] = x_new
        
    # Return memory into single vector
    def get(self):
        return self.memory.reshape(-1, 1, order='F')
    
    # Unify both functionalities into single one
    def add_and_get(self, x):
        self.add(x)
        return self.memory.reshape(-1, 1, order='F')
