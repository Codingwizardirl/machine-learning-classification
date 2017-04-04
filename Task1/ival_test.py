import numpy as np
from visualize_and_preprocess import train_x
from MyKmeans import MyKmeans
import matplotlib.pyplot as plt

k = 8
length = train_x.shape[1]
NUMBER_OF_TESTS = 30
SSE_List = []
#   Run 30 K-Means cluster iterations with random initial centres
for i in range(NUMBER_OF_TESTS):
    centres = np.random.rand(k, length)
    C, idx, SSE = MyKmeans(train_x, k, centres)
    sse = SSE[SSE.shape[0] - 1]
    SSE_List.append(sse)

# Save the result
np.savetxt('SSE_List.out', SSE_List)

SSE_List = np.loadtxt('SSE_List.out')

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