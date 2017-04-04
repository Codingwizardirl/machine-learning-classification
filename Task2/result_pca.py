import scipy.io
from compute_pca import compute_pca
from Task1.visualize_and_preprocess import train_x

# Find Evecs and Evals for the training data
EVecs, EVals = compute_pca(train_x)

# Produce .mat files for coursework
scipy.io.savemat('evecs.mat', {'Eigenvectors':EVecs[:,0:10]})
scipy.io.savemat('evals.mat', {'Eigenvalues':EVals[0:10]})

# Save all Evecs for later use
# np.savetxt('allEvecs.out', EVecs)