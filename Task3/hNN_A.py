import numpy as np
from hNeuron import hNeuron

def hNN_A(X):
    # Task 3.4
    #  X: data matrix of N-by-D
    #  Y: output vector of N-by-1
    Z1 = np.array([-3.325, -0.05, 1])
    Z2 = np.array([8.5, -1, 0])
    Z3 = np.array([303.0/56, 11.0/28, -1])
    Z4 = np.array([-7.875, 1.25, 1])
    Z5 = np.array([-3.5, 1, 1, 1, 1])

    Y1 = hNeuron(Z1, X)
    Y2 = hNeuron(Z2, X)
    Y3 = hNeuron(Z3, X)
    Y4 = hNeuron(Z4, X)

    X_new = np.vstack((Y1,Y2,Y3,Y4))
    Y = hNeuron(Z5, X_new)

    return Y


X = np.array([[3.5, 3.5], [8.5,3.75],[8.5, 8.75],[1.5,6]])

