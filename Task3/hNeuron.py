import numpy as np

def hNeuron(W, X):
    # Task 3.1
    #  X: data matrix of N-by-D
    #  W: weight vector of (D+1)-by-1
    #  Y: output vector of N-by-1
    bias = W[0]
    W = np.delete(W, 0)
    a = W.T.dot(X) + bias
    Y = (a > 0).astype(int)
    return Y