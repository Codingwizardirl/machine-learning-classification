import numpy as np
from hNeuron import hNeuron

def hNN_AB(X):
    # Task 3.6
    #  X: data matrix of N-by-D
    #  Y: output vector of N-by-1

    # Weight vectors for first layer perceptrons
    Z1 = np.array([-3.325, -0.05, 1])
    Z2 = np.array([8.5, -1, 0])
    Z3 = np.array([303.0 / 56, 11.0 / 28, -1])
    Z4 = np.array([-7.875, 1.25, 1])

    Z5 = np.array([-4.75, 1, 0])
    Z6 = np.array([6.25, -1, 0])
    Z7 = np.array([-5, 0, 1])
    Z8 = np.array([6, 0, -1])

    #  Weight vector for second layer perceptron
    Z9 = np.array([-3.5, 1, 1, 1, 1])

    # Outputs from first layer for shape A
    Y1 = hNeuron(Z1, X)
    Y2 = hNeuron(Z2, X)
    Y3 = hNeuron(Z3, X)
    Y4 = hNeuron(Z4, X)

    # Outputs from first layer for shape B
    Y5 = hNeuron(Z5, X)
    Y6 = hNeuron(Z6, X)
    Y7 = hNeuron(Z7, X)
    Y8 = hNeuron(Z8, X)

    # Weights for logical operation neurons
    And= np.array([-1.5, 1, 1])
    Not = np.array([0, -1])

    X_A = np.vstack((Y1, Y2, Y3, Y4))
    X_B = np.vstack((Y5, Y6, Y7, Y8))
    # Result of points being in A
    Y_A = hNeuron(Z9, X_A)
    # Result of points being in B
    Y_B = hNeuron(Z9, X_B)
    # Result of points being outside of B
    Y_B = hNeuron(Not, Y_B[np.newaxis,:])

    Y_AB = np.vstack((Y_A,Y_B))
    Y = hNeuron(And, Y_AB)

    return Y
