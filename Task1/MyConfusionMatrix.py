from visualize_and_preprocess import train_x, train_y
import numpy as np

def MyConfusionMatrix(y_actual, y_predicted):
    '''
    
    :param y_actual: actual labels
    :param y_predicted: predicted labels of data
    :return: Confusion matrix for the data as a 2d numpy array
    '''
    if(y_actual.shape[0] != y_predicted.shape[0]):
        raise ValueError("Vectors supplied must be of same length")

    # Minimum and maximum values in order to know the size of the confusion matrix and how to access it
    max_actual = np.max(y_actual)
    max_predicted = np.max(y_predicted)
    min_actual = np.min(y_actual)
    min_predicted = np.min(y_predicted)

    N = max_actual + 1 if min_actual == 0 else max_actual
    M = max_predicted + 1 if min_predicted == 0 else max_predicted

    confusion_matrix = np.zeros((N,M))

    length = y_predicted.shape[0]

    for index in range(length):
        i = y_actual[index] if min_actual == 0 else y_actual[index] - 1
        j = y_predicted[index] if min_predicted == 0 else y_predicted[index] - 1
        confusion_matrix[i,j] += 1

    return confusion_matrix
