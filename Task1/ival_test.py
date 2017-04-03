import numpy as np
from visualize_and_preprocess import train_x
from MyKmeans import MyKmeans
import matplotlib.pyplot as plt

# k = 8
# length = train_x.shape[1]
# NUMBER_OF_TESTS = 30
# SSE_List = []
#
# for i in range(NUMBER_OF_TESTS):
#     centres = np.random.rand(k, length)
#     C, idx, SSE = MyKmeans(train_x, k, centres)
#     sse = SSE[SSE.shape[0] - 1]
#     SSE_List.append(sse)

# np.savetxt('SSE_List.out', SSE_List)

SSE_List = np.loadtxt('SSE_List.out')
print np.max(SSE_List)
print np.min(SSE_List)
print np.max(SSE_List) - np.min(SSE_List)
# Make a plot for the data
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Iteration', fontsize=20)
plt.ylabel('Squared Sum Error Value', fontsize=20)
plt.plot(SSE_List)
plt.figure()
# Plot histogram
plt.hist(SSE_List, histtype='stepfilled', bins=10)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel('Squared Sum Error Value', fontsize=20)
plt.ylabel('Iterations', fontsize=20)
plt.show()