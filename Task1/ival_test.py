import numpy as np
from visualize_and_preprocess import train_x
from MyKmeans import MyKmeans
import matplotlib.pyplot as plt

k = 8
length = train_x.shape[1]
NUMBER_OF_TESTS = 30
SSE_List = []

# for i in range(NUMBER_OF_TESTS):
#     centres = np.random.rand(k, length)
#     C, idx, SSE = MyKmeans(train_x, k, centres)
#     sse = SSE[SSE.shape[0] - 1]
#     SSE_List.append(sse)
#
# np.savetxt('test.out', SSE_List, delimiter=',')

SSE_List = np.loadtxt('test.out')

plt.plot(SSE_List)
plt.xlabel('Iteration')
plt.ylabel('Square Sum Error Value')
plt.show()