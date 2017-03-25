import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Loading data (Substitute UUN)
train_mat = scipy.io.loadmat('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/UUN/train_data.mat')
test_mat = scipy.io.loadmat('/afs/inf.ed.ac.uk/group/teaching/inf2b/cwk2/d/UUN/test_data.mat')
train_x = np.array(train_mat['train_x'])
train_y = np.array(train_mat['train_y'])
test_x = np.array(test_mat['test_x'])
test_y = np.array(test_mat['test_y'])

train_x = train_x/256.0
test_x = test_x/256.0

# Visualise images
plt.imshow(train_x[0,:,:,:])
plt.show()

# Pre-process data
train_x = np.reshape(train_x,(train_x.shape[0],-1), order = 'F')
test_x = np.reshape(test_x, (test_x.shape[0],-1), order = 'F')

