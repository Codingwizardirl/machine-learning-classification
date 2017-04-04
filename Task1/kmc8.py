import numpy as np
import scipy.io
import pandas as pd
from MyKmeans import MyKmeans
from visualize_and_preprocess import train_x, train_y
from MyConfusionMatrix import MyConfusionMatrix

# Pick number of clusters and initial centres
k = 8
centres = train_x[0:k]

# Perform K-means clustering with 8 clusters assigned to first 8 points of the data
C, idx, SSE = MyKmeans(train_x, k, centres)
sse = SSE[SSE.shape[0]-1]
print ('Final SSE: %0.6f' % (sse))

np.savetxt('SSE.out', SSE)
SSE = np.loadtxt('SSE.out')

scipy.io.savemat('result_C.mat', {'centres':C})

# Calculate the confusion matrix and save it in .mat file
confusion_matrix = MyConfusionMatrix(train_y, idx)
scipy.io.savemat('result_confusion_matrix.mat', {'confusion_matrix': confusion_matrix})

# Pretty printing for the confusion matrix using Pandas. Used only for the report.
# y_actual = pd.Index(np.arange(1,6,1), name="Actual")
# y_pred = pd.Index(np.arange(1,9,1), name="Predicted")
# df = pd.DataFrame(data = confusion_matrix.astype(int), index = y_actual, columns=y_pred)
# print df
