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

scipy.io.savemat('result_C.mat', {'centres':C})

# Calculate the confusion matrix
confusion_matrix = MyConfusionMatrix(train_y, idx)
scipy.io.savemat('result_confusion_matrix.mat', {'confusion_matrix': confusion_matrix})

# Pandas confusion matrix
# y_actu = pd.Series(train_y.ravel(), name='Actual')
# y_pred = pd.Series(idx.ravel(), name='Predicted')
# df_confusion = pd.crosstab(y_actu, y_pred)
# print df_confusion