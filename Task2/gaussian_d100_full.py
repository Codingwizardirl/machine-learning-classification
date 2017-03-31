from compute_pca import my_cov, compute_pca
import numpy as np
from Task1.MyKmeans import my_mean
from Task1.visualize_and_preprocess import train_x, train_y, test_x


def gaussianClassifier(mu, covar, X):
    '''
    
    :param X: Data
    :return: 
    '''


    n,d = X.shape
    j,k = covar.shape
    if(j != d or k != d):
        raise ValueError("Dimension of covariance matrix and data should match")

    inverse_cov = np.linalg.inv(covar)
    mu = mu.reshape((1,d))

    X = X - np.ones((n,1)).dot(mu)
    fact = np.sum((X.dot(inverse_cov)*X), 1)

    p = np.exp(-0.5*fact)
    p = p/np.sqrt((pow(2*np.pi,d))*np.linalg.det(covar))
    return p



EVecs = np.loadtxt('evecs100.out')
X = train_x.dot(EVecs)
covar = my_cov(X)
mu = my_mean(X, 0)



classes = 5
mu_hat = np.zeros((100,classes))
sigma_hat = np.zeros((100,100,classes))

for k in range(classes):
    print X[1,:].shape
    print X[np.where(train_y-1== k),:].shape
    # mu_hat[:,k] = my_mean(X[np.where((train_y-1)== k)], 0)
    # sigma_hat[:,:,k] = my_cov(X[np.where((train_y-1)== k)])
# test = test_x.dot(EVecs[:,0:100])
# probabilities = np.zeros((classes, train_y.shape[0], 1))
#
# for k in range(classes):
#     probabilities[k,...] = gaussianClassifier(mu)



