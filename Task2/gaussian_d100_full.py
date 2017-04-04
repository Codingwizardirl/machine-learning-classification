# For module imports from other tasks
import sys
sys.path.append("..")

from compute_pca import my_cov
import numpy as np
import scipy.io
from Task1.MyKmeans import my_mean
from Task1.visualize_and_preprocess import train_x, train_y, test_x, test_y
from Task1.MyConfusionMatrix import MyConfusionMatrix
from compute_pca import compute_pca


def gaussianClassifier(mu, covar, X):
    '''
    Return posterior probabilities approximated by a Gaussian with provided mean and covariance.
     :param mu: Mean vector of the data
     :param covar: Covariance matrix of the data
     :param X: Data to be classified
     :return: p: posterior probabilities
    '''

    n,d = X.shape
    j,k = covar.shape
    if(j != d or k != d):
        raise ValueError("Dimension of covariance matrix and data should match")

    inverse_cov = np.linalg.inv(covar)
    mu = mu.reshape((1,d))

    X = X - mu
    fact = np.sum((X.dot(inverse_cov)*X), 1)

    p = np.exp(-0.5*fact)
    p = p/np.sqrt((pow(2*np.pi,d))*np.linalg.det(covar))
    return p


def trainAndTest(train_x, train_y, test_x, test_y, EVecs, dimensionality):
    '''
    Fit model with the training data and then compute confusion matrix for the test data.
    :param train_x: Training data to fit the Gaussian model on 
    :param train_y: Labels of the training data
    :param test_x:  Test data to classify
    :param test_y:  Correct labels
    :param EVecs:   Evecs for the training data - so they don't have to be recomputed on every run of the experiments
    :param dimensionality Dimensionality for the data to be reduced to
    :return: Confusion matrix for the test data
    '''

    # Reduce dimensionality and center of training data using the mean
    EVecs = EVecs[:,0:dimensionality]
    mu = my_mean(train_x, 0)
    X = train_x - mu
    X = X.dot(EVecs)

    # Variables to hold the means, covariance matrices and their determinants
    classes = 5
    mu_hat = np.zeros((classes,dimensionality))
    sigma_hat = np.zeros((dimensionality,dimensionality,classes))
    determinants = np.zeros(classes)

    # Populate the variables above
    for k in range(classes):
        idx = np.ravel((train_y-1) == k)
        mu_hat[k,:] = my_mean(X[idx,:], 0)
        sigma_hat[:,:,k] = my_cov(X[idx,:])
        determinants[k] = np.linalg.det(sigma_hat[:,:,k])
        # print determinants[k]

    # Reduce dimensionality of test data and center it using the training mean
    T = test_x - mu
    T = T.dot(EVecs)
    probabilities = np.zeros((classes, test_y.shape[0]))

    # Find posterior probabilities for every class
    for k in range(classes):
        probabilities[k,:] = gaussianClassifier(mu_hat[k,:], sigma_hat[:,:,k], T)

    # Find predicted class labels for the test data and correct it so it starts from 1
    labels_predicted = np.argmax(probabilities,axis=0) + 1
    confusion_matrix = MyConfusionMatrix(test_y, labels_predicted)
    return confusion_matrix

EVecs, EVals = compute_pca(train_x)
confusion_matrix = trainAndTest(train_x,train_y, test_x, test_y, EVecs, 100)
scipy.io.savemat('confmat_d100.mat', {'Confusion Matrix': confusion_matrix})
classification_rate = np.diag(confusion_matrix / confusion_matrix.sum(axis =0))

for float in classification_rate:
    print '{:.1%}'.format(float)
    
# Pretty printing for the confusion matrix using Pandas. Used only for the report.
# y_actual = pd.Index(np.arange(1,6,1), name="Actual")
# y_pred = pd.Index(np.arange(1,6,1), name="Predicted")
# df = pd.DataFrame(data = confusion_matrix.astype(int), index = y_actual, columns=y_pred)
# print df


