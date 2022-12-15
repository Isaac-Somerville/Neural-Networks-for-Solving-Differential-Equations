#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt


# Data for a three-dimensional line
x = np.linspace(0,1,20)
y = np.linspace(0,1,20)
X,Y = np.meshgrid(x,y)
Z = np.exp(-X) * (X+Y**3)
ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(30, 315)


# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')