import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sNN_AB import sNN_AB

# Generate points between 0 and 10 to classify
xplot = np.linspace(0,10, 2000)
yplot = np.linspace(0,10,2000)
xx, yy = np.meshgrid(xplot, yplot)
gridX = np.vstack((xx.ravel(), yy.ravel()))

# Classify the points and reshape the result to fit the plot function.
data = sNN_AB(gridX)
data = data.reshape((xplot.shape[0], yplot.shape[0]))

# Setup the plot title and axis
plt.title('Task 3.9 Plot')
plt.xticks(np.arange(0, 10, 0.5), fontsize=16)
plt.yticks(np.arange(0, 10, 0.5),fontsize=16)
plt.xlabel('x1', fontsize=22)
plt.ylabel('x2', fontsize=22)


# Plot data and show result
plt.contourf(xx, yy, data, cmap=cm.Greys)

plt.show()