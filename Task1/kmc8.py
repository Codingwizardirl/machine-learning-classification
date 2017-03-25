import numpy as np
import scipy.io
from MyKmeans import MyKmeans
from visualize_and_preprocess import train_x

k = 8
centres = train_x[0:k]
C, idx, SSE = MyKmeans(train_x, k, centres)
print SSE
# scipy.io.savemat('result_C.mat', C)