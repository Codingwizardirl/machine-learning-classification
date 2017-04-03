import numpy as np
from sNeuron import sNeuron
def sNN_AB(X):
    # Task 3.8
    #  X: data matrix of N-by-D
        #  Y: output vector of N-by-1

    #  multiplierA: scalar to apply to all weight vectors for PolygonA
    #  multiplierB: scalar to apply to all weight vectors for PolygonB
    multiplierA = 50
    multiplierB = 30

    # Weight vectors for first layer perceptrons for
    Z1 = np.array([-3.325, -0.05, 1])*multiplierA
    Z2 = np.array([8.5, -1, 0])*multiplierA
    Z3 = np.array([303.0 / 56, 11.0 / 28, -1])*multiplierA
    Z4 = np.array([-7.875, 1.25, 1])*multiplierA

    Z5 = np.array([-4.75, 1, 0])*multiplierB
    Z6 = np.array([6.25, -1, 0])*multiplierB
    Z7 = np.array([-5, 0, 1])*multiplierB
    Z8 = np.array([6, 0, -1])*multiplierB


    #  Weight vector for second layer perceptron
    Z9 = np.array([-3.5, 1, 1, 1, 1])*multiplierB

    # Outputs from first layer for shape A
    Y1 = sNeuron(Z1, X)
    Y2 = sNeuron(Z2, X)
    Y3 = sNeuron(Z3, X)
    Y4 = sNeuron(Z4, X)

    # Outputs from first layer for shape B
    Y5 = sNeuron(Z5, X)
    Y6 = sNeuron(Z6, X)
    Y7 = sNeuron(Z7, X)
    Y8 = sNeuron(Z8, X)

    # Weights for logical operation neurons
    And = np.array([-1.5, 1, 1])
    Not = np.array([0, -1])

    X_A = np.vstack((Y1, Y2, Y3, Y4))
    X_B = np.vstack((Y5, Y6, Y7, Y8))
    # Result of points being in A
    Y_A = sNeuron(Z9, X_A)
    # Result of points being in B
    Y_B = sNeuron(Z9, X_B)
    # Result of points being outside of B
    Y_B = sNeuron(Not, Y_B[np.newaxis, :])

    Y_AB = np.vstack((Y_A, Y_B))
    Y = sNeuron(And, Y_AB)

    return Y
