#%%
from neurodiffeq import diff
from neurodiffeq.solvers import Solver1D, Solver2D
from neurodiffeq.conditions import IVP, DirichletBVP2D
from neurodiffeq.networks import FCNN, Swish
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import chain

def ode_system(u,v,t):
    return [diff(u,t) - torch.cos(t) - u**2 - v + (1 + t**2 + torch.sin(t)**2),
            diff(v,t) - 2*t + (1+t**2)*torch.sin(t) - u*v]
    
conditions = [IVP(t_0 = 0.0, u_0 = 0.0), IVP(t_0 = 0.0, u_0 = 1.0)]
nets = [FCNN(n_hidden_units=10), FCNN(n_hidden_units=10)]
params = set(chain.from_iterable(n.parameters() for n in nets))

solver = Solver1D(ode_system, conditions, t_min=0.0, t_max=3.0, 
                  nets=nets, optimizer = torch.optim.Adam(params, lr = 1e-2), 
                  criterion=torch.nn.MSELoss(), batch_size=30)
solver.fit(max_epochs=10000)
solution = solver.get_solution()

t = np.linspace(0,3,30)
sol1 = np.sin(t)
sol2 = 1 + t**2
u, v = solution(t, to_numpy=True)

plt.plot(t, u, 'r-', label = "Neural Network 1 Output")
plt.plot(t, sol1, 'b.', label = "True Solution")

plt.xlabel("t")
plt.ylabel("u(t)")
plt.legend(loc = "lower center")
plt.show()

plt.plot(t, v, 'r-', label = "Neural Network 2 Output")
plt.plot(t, sol2, 'b.', label = "True Solution")

plt.xlabel("t")
plt.ylabel("v(t)")
plt.legend(loc = "lower center")
plt.show()