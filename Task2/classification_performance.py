# For module imports from other tasks
import sys
sys.path.append("..")

import numpy as np
from Task1.visualize_and_preprocess import train_x, train_y, test_x, test_y
from gaussian_d100_full import trainAndTest
from compute_pca import compute_pca
import matplotlib.pyplot as plt

# Compute EVecs for the training data
EVecs, EVals = compute_pca(train_x)

# Run tests with the following dimensions and report the classification rate
dimensions = [2, 10, 20, 30, 50, 75, 100, 125, 150, 250]
average_rates = []
for dimensionality in dimensions:
    confusion_matrix = trainAndTest(train_x,train_y, test_x, test_y, EVecs, dimensionality)
    classification_rate = np.diag(confusion_matrix / confusion_matrix.sum(axis =0))

    # Print classification rate for the 5 classses in %
    print ('Classification rate for data with dimensionality reduced to: %d' % (dimensionality))
    for float in classification_rate:
        print '{:.1%}'.format(float)

    # Report average classification rate
    average_rate = classification_rate.sum()/classification_rate.shape[0]
    # Store average rate in % to plot later
    average_rates.append(average_rate*100)
    print ('Average classification rate: %.2f%% \n' % (average_rate*100))

# Plot average classification rates against dimensions
plt.xticks(fontsize=20)
plt.yticks(np.arange(30,100, 5), fontsize=20)
plt.xlabel('Dimensions of data', fontsize=20)
plt.ylabel('Average classification rate (%)', fontsize=20)
plt.plot(dimensions, average_rates)
plt.show()