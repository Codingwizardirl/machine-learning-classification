import numpy as np
from Task1.visualize_and_preprocess import train_x, train_y, test_x, test_y
from gaussian_d100_full import trainAndTest


dimensions = [2, 10, 20, 30, 50, 75, 100, 125, 150, 250]
for dimensionality in dimensions:
    confusion_matrix = trainAndTest(train_x,train_y, test_x, test_y, dimensionality)
    classification_rate = np.diag(confusion_matrix / confusion_matrix.sum(axis =0))
    # Print classification rate in %
    print ('Classification rate for data with dimensionality reduced to: %d' % (dimensionality))
    for float in classification_rate:
        print '{:.1%}'.format(float)
    average_rate = classification_rate.sum()/classification_rate.shape[0]
    print ('Average classification rate: %.2f%% \n' % (average_rate*100))