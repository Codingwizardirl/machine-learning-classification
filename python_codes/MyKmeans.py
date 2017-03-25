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

	# TO-DO
	C = None
	idx = None
	SSE = None
	raise NotImplementedError('Kmeans function not implemented')

	return C, idx, SSE
