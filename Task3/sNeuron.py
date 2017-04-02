import numpy as np

def sNeuron(W,X):
    # Task 3.2
    #  X: data matrix of N-by-D
    #  W: weight vector of (D+1)-by-1
    #  Y: output vector of N-by-1
    a = W.T.dot(X)
    Y = 1.0 / (1 + np.exp(-1.0*a))
    return Y