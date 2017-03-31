import numpy as np
from Task1.MyKmeans import my_mean

def compute_pca(X):

    '''
        Input:
            X:			Input sample data, N x D matrix
        Output:
            EVecs:		A matrix contains all eigenvectors as columns, D x D matrix
            EVals:		Eigenvalues in descending order, D x 1 vector
            (Note that the i-th columns of Evecs should corresponds to the i-th element in EVals)
    '''

    # TO-DO
    EVecs = None
    EVals = None

    covariance_matrix = my_cov(X)
    (EVals, EVecs) = np.linalg.eig(covariance_matrix)

    # Sort the EVals and EVects in decreasing order
    sorted_order = np.argsort(EVals)
    sorted_order = sorted_order[::-1]
    EVals = EVals[sorted_order]
    EVecs = EVecs[...,sorted_order]

    # Check for first negative element of vectors
    for i in range(X.shape[1]):
        if EVecs[0,i] < 0:
            EVecs [0,i] *= -1

    return EVecs, EVals

def my_cov(X):
    '''
    
    :param X: Input sample data to calculate covariance matrix on
    :return: A covariance matrix for the sample data
    '''
    N = X.shape[0]
    X_mean = my_mean(X, 0)
    X = X - X_mean
    covariance_matrix = (1.0/N) * np.dot(X.T, X)

    return covariance_matrix
