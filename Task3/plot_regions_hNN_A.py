import numpy as np
import matplotlib.pyplot as plt
from hNN_A import hNN_A

xplot = np.linspace(0,10, 2000)
yplot = np.linspace(0,10,2000)
print xplot.shape
xx, yy = np.meshgrid(xplot, yplot)

gridX = np.vstack((xx.ravel(), yy.ravel()))

data = hNN_A(gridX)
data = data.reshape((xplot.shape[0], yplot.shape[0]))
plt.xticks(np.arange(0, 10, 0.5))
plt.yticks(np.arange(0, 10, 0.5))
plt.contourf(xx, yy, data)

plt.show()

