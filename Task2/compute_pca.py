import numpy as np
from Task1.MyKmeans import my_mean
from Task1.visualize_and_preprocess import train_x

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
    cv2 = np.cov(X,rowvar=False)
    print covariance_matrix
    print cv2
    return EVecs, EVals

def my_cov(X):
    '''
    
    :param X: Input sample data to calculate covariance matrix on
    :return: A covariance matrix for the sample data
    '''
    N = X.shape[0]
    X_mean = my_mean(X, 0)
    X = X - X_mean
    covariance_matrix = (1.0/(N-1)) * np.dot(X.T, X)

    return covariance_matrix

x = np.array([[2.5, 2], [0.5, 0.7], [2.2, 2.4], [1.9,2.5],[3.1,3.9],[2.3,2.7],[2,1.8],[1.5,1.6],[1, 1.3]])
compute_pca(x)