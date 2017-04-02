import numpy as np

def sNeuron(W,X):
    a = W.T.dot(X)
    result_vector = 1.0 / (1 + np.exp(-1.0*a))