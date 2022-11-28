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


"""
Scaled ODE with custom parameters to match my own attempt
"""
def ode_system(u,v,t):
    return [diff(u,t) - 3*(torch.cos(3*t) + u**2 + v - (1 + (3*t)**2 + torch.sin(3*t)**2)),
            diff(v,t) - 3*(2*3*t - (1+(3*t)**2)*torch.sin(3*t) + u*v)]
    
# conditions = [IVP(t_0 = 0.0, u_0 = 0.0), IVP(t_0 = 0.0, u_0 = 1.0)]
# nets = [FCNN(hidden_units=(32,32)), FCNN(hidden_units=(32,32))]
# params = set(chain.from_iterable(n.parameters() for n in nets))

# solver = Solver1D(ode_system, conditions, t_min=0.0, t_max=1.0, 
#                   nets=nets, optimizer = torch.optim.Adam(params, lr = 1e-2), 
#                   criterion=torch.nn.MSELoss(), train_generator=Generator1D(30, t_min=0, t_max=1, method='uniform'))
# solver.fit(max_epochs=10000)
# solution = solver.get_solution()

t = np.linspace(0,1,30)
sol1 = np.sin(3*t)
sol2 = 1 + (3*t)**2
# u, v = solution(t, to_numpy=True)

"""
Unscaled ODE with default params
2 hidden layers, 32 hidden units in each
32 Points equally spaced with noise, generated every epoch
Adam optimizer, lr 0.001
loss function ?
"""
# def ode_system(u,v,t):
#     return [diff(u,t) - (torch.cos(t) + u**2 + v - (1 + (t)**2 + torch.sin(t)**2)),
#             diff(v,t) - (2*t - (1+(t)**2)*torch.sin(t) + u*v)]
    
conditions = [IVP(t_0 = 0.0, u_0 = 0.0), IVP(t_0 = 0.0, u_0 = 1.0)]
nets = [FCNN(), FCNN()]


solver = Solver1D(ode_system, conditions, t_min=0.0, t_max=1.0, nets=nets)
solver.fit(max_epochs=5000)
solution = solver.get_solution()

# t = np.linspace(0,3,30)
# sol1 = np.sin(t)
# sol2 = 1 + (t)**2
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
plt.legend(loc = "upper left")
plt.show()
# %%
