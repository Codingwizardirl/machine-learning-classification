import numpy as np
from hNeuron import hNeuron

def hNN_A(X):
    # Task 3.4
    #  X: data matrix of N-by-D
    #  Y: output vector of N-by-1
    Z1 = np.array([-133/40, -1/20, 1]).astype(float)
    Z2 = np.array([-17/2, 1, 0]).astype(float)
    Z3 = np.array([303/56, 11/28, -1]).astype(float)
    Z4 = np.array([-63/8, 5/4, 1]).astype(float)
    Z5 = np.array([-7/2, 1, 1, 1, 1]).astype(float)

    Y1 = hNeuron(Z1, X)
    Y2 = hNeuron(Z2, X)
    Y3 = hNeuron(Z3, X)
    Y4 = hNeuron(Z4, X)

    X_new = np.column_stack((Y1,Y2,Y3,Y4))
    Y = hNeuron(Z5, X_new)

    return Y

