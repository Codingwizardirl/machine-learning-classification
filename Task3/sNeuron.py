import numpy as np

def sNeuron(W,X):
    # Task 3.2
    #  X: data matrix of N-by-D
    #  W: weight vector of (D+1)-by-1
    #  Y: output vector of N-by-1
    bias = W[0]
    W = np.delete(W, 0)
    a = np.dot(W.T,X) + bias
    Y = 1.0 / (1 + np.exp(-1.0*a))
    return Y