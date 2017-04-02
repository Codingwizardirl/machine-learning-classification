import numpy as np

def hNeuron(W, X):
    a = W.T.dot(X)
    result_vector = (a > 0).astype(int)
    return result_vector