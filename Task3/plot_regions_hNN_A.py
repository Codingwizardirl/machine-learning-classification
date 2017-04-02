import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from hNN_A import hNN_A

# Generate points between 0 and 10 to classify
xplot = np.linspace(0,10, 2000)
yplot = np.linspace(0,10,2000)
xx, yy = np.meshgrid(xplot, yplot)
gridX = np.vstack((xx.ravel(), yy.ravel()))

# Classify the points and reshape the result to fit the plot function
data = hNN_A(gridX)
data = data.reshape((xplot.shape[0], yplot.shape[0]))

# Setup the plot title and axis
plt.title('Task 3.5 Plot')
plt.xticks(np.arange(0, 10, 0.5), fontsize=16)
plt.yticks(np.arange(0, 10, 0.5),fontsize=16)
plt.xlabel('x1', fontsize=22)
plt.ylabel('x2', fontsize=22)

# Setup legend

white_patch_legend = mpatches.Patch(color='white', label='Class 0')
black_patch_legend = mpatches.Patch(color='black', label='Class 1')
plt.legend(loc='best', fancybox=True, framealpha=0.2, handles=[white_patch_legend, black_patch_legend], facecolor='black', fontsize=16)

# Plot data and show result
plt.contourf(xx, yy, data, cmap=cm.Greys)
plt.show()
