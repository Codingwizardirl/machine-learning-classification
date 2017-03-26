import numpy as np
import scipy.io
import pandas as pd
from MyKmeans import MyKmeans
from visualize_and_preprocess import train_x, train_y
from MyConfusionMatrix import MyConfusionMatrix

# Perform K-means clustering with 8 clusters assigned to first 8 points of the data
k = 8
centres = train_x[0:k]

C, idx, SSE = MyKmeans(train_x, k, centres)
scipy.io.savemat('result_C.mat', {'centres':C})

sse = SSE[SSE.shape[0]-1]

confusion_matrix = MyConfusionMatrix(train_y, idx)
scipy.io.savemat('result_confusion_matrix.mat', {'confusion_matrix': confusion_matrix})

# Format data from K-means clustering into pandas series for confusion matrix calculation
# y_actual = pd.Series(np.ravel(train_y), name='Actual')
# y_clustered = pd.Series(np.ravel(idx), name='Predicted')
# confusion_matrix = pd.crosstab(y_actual, y_clustered)
# conf_dict = {col_name : confusion_matrix[col_name].values for col_name in confusion_matrix.columns.values}
# scipy.io.savemat('result_confusion_matrix.mat', {'confusion_matrix': conf_dict})