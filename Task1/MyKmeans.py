import numpy as np

def MyKmeans(X,k,initialCentres,maxIter = 500):

    '''
        Input:
            X:				Input sample data, N x D matrix
            k:				Number of target clusters, integer
            initialCentres:	Initial k cluster centers, k x D matrix
            maxIter:		Maximum number of iteration, integer
        Output:
            C:		Final cluster centres, k x D matrix
            idx:			Cluster index table of samples, N x 1 vector
            SSE:			Sum-squared error for each Iteration, where sse[0] corresponds
                            to the error of initial centres, and sse[i] corresponds to
                            error after i iterations, (L+1) x 1 vector
    '''

    N = X.shape[0]
    D = np.zeros((k, N))
    C = initialCentres
    idx_previous = np.zeros((1, N))
    SSE = []

    for i in range(maxIter):
        for c in range(k):
             D[c] = square_dist(X, C[c])

        idx = np.argmin(D, axis=0)
        SSE.append(sumSquareError(X, C, idx))

     #    Update clusters
        for c in range(k):
            if(np.sum(idx == c) == 0):
                print("WARNING: Empty cluster")
            else:
                C[c] = my_mean(X[idx == c], 0)

        if( np.sum(np.abs(idx_previous - idx)) == 0):
            break

        idx_previous = idx

    SSE = np.array(SSE)
    return C, idx, SSE


def square_dist(U, v):
    sq_dist = np.sum(((U-v)**2), axis=1).T
    return sq_dist

def my_mean(U, ax = None):
    # Returns mean column vector of a given matrix U
    N = U.shape[0]
    return (1.0/N) * np.sum(U, axis=ax)

def sumSquareError(U, clusters, idx):
    sse = 0
    N = U.shape[0]
    for c in range(clusters.shape[0]):
        sse += np.sum((U[idx == c] - clusters[c])**2)
    sse /= N
    return sse
