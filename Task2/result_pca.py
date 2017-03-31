import scipy.io
import numpy as np
from compute_pca import compute_pca
from Task1.visualize_and_preprocess import train_x

EVecs, EVals = compute_pca(train_x)
np.savetxt('allEvecs.out', EVecs)
scipy.io.savemat('evecs.mat', {'Eigenvectors':EVecs[:,0:10]})
scipy.io.savemat('evals.mat', {'Eigenvalues':EVals[0:10]})