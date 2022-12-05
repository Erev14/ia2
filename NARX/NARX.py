import numpy as np
from MLP import *
from DelayBlock import *

class NARX:
    # Non linear AutoRegressor with eXogeneous inputs.
    # With multi layer neural network as non linear funtion    
    def __init__(self, n_inputs, n_outputs, n_delays,
                 dense_hidden_layers=(100,),
                 learning_rate=0.01,
                 n_repeat_train=5):
        
        self.net = MLP(((n_inputs+n_outputs)*n_delays,
                        *dense_hidden_layers, n_outputs),
                       output_activation='linear')
        
        self.dbx = DelayBlock(n_inputs, n_delays)
        self.dby = DelayBlock(n_outputs, n_delays)
        self.learning_rate = learning_rate
        self.n_repeat_train = n_repeat_train
        
    # NARX prediction
    # x is the vector size (n_inputs,1)
    def predict(self, x):
        # Prepare the input extended over time
        X_block = self.dbx.add_and_get(x)
        Y_est_block = self.dby.get()
        net_input = np.vstack((X_block, Y_est_block))
        
        # Neural network prediction
        y_est = self.net.predict(net_input)
        
        # save prediction on the recurrent block
        self.dby.add(y_est)
        
        # Return prediction
        return y_est
    
    # Prediction and training in NARX
    # X: input vector of size (n_outputs, 1)
    # y: output vector of size (n_outputs, 1)
    def predict_and_train(self, x, y):
        # the prediction is given before training but is not saved before training
        X_block = self.dbx.add_and_get(x)
        Y_est_block = self.dby.get()
        net_input = np.vstack((X_block, Y_est_block))
        
        # Neural network prediction
        y_est = self.net.predict(net_input)
        
        # neural network train 
        self.net.fit(net_input, y, 
                       epochs=self.n_repeat_train,
                       learning_rate = self.learning_rate)
        
        # Save prediction in the recurrent block
        self.dby.add(y_est)
        
        # Return prediction
        return y_est