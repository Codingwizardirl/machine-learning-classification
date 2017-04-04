import numpy as np

def hNeuron(W, X):
    '''
    Task 3.1
    Function immitating functionality of neuron using step function
    :param W: weight vector of (D+1)-by-1
    :param X: data matrix of D-by-N
    :return: Y: output vector of N-by-1
    '''
    bias = W[0]
    W = np.delete(W, 0)
    a = np.dot(W.T,X) + bias
    Y = (a >= 0).astype(int)
    return Y

