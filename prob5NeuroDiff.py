#%%
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, Swish, SinActv
from neurodiffeq.generators import Generator1D
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def pde_system(u, x, y):
    return [diff(u, x, order=2) + diff(u, y, order=2) - torch.exp(-x)*(x-2+y**3+6*y)]

conditions = [
    DirichletBVP2D(
        x_min=0, x_min_val=lambda y: y**3,
        x_max=1, x_max_val=lambda y: (1+y**3)*np.exp(-1),                   
        y_min=0, y_min_val=lambda x: x*torch.exp(-x),                   
        y_max=1, y_max_val=lambda x: torch.exp(-x)*(x+1),                   
    )
]
nets = [FCNN(n_input_units=2, n_output_units=1, hidden_units=(32,))]

solver = Solver2D(pde_system, conditions, xy_min=(0, 0), xy_max=(1, 1), nets=nets)
solver.fit(max_epochs=2000)
solution = solver.get_solution()


x = np.linspace(0,1,10)
y = np.linspace(0,1,10)

X, Y = np.meshgrid(x,y)
u = solution(X, Y, to_numpy=True)
soln = np.exp(-X) * (X + Y**3)


ax = plt.axes(projection='3d')
ax.plot_surface(X,Y,u,rstride=1, cstride=1,
            cmap='plasma', edgecolor='none')
ax.scatter(X,Y,soln)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc = "lower center")
ax.view_init(30, 315)
plt.show()