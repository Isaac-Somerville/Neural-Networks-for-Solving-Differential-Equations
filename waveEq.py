#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# maybe jacobian instead?

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and t-coordinates as test data"""
    def __init__(self, xrange, trange, num_samples):
        X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
        T  = torch.linspace(trange[0],trange[1],num_samples, requires_grad=True)
        # create tuple of (num_samples x num_samples) points
        x,t = torch.meshgrid(X,T) 

        # input of forward function must have shape (batch_size, 2)
        self.data_in = torch.cat((x.reshape(-1,1),t.reshape(-1,1)),1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, i):
        return self.data_in[i]

class Fitter(torch.nn.Module):
    """Forward propagations"""
    def __init__(self, num_hidden_nodes):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(2, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, input):
        hidden = torch.sigmoid(self.fc1(input))
        z = self.fc2(hidden)
        return z

class DiffEq():
    """
    The wave equation with fixed end points

    u_{tt} - v^2 * u_{xx} = 0, 0 < x < L , t > 0 
    u(0,t) = u(L,t) = 0
    u(x,0) = f(x), u_{t}(x,0) = g(x)

    """
    def __init__(self, xrange, trange, L, v):
        self.xrange = xrange
        self.trange = trange
        self.L = L
        self.v = v

    def integrate(self,func,n, num_points):
        x = torch.linspace(0,self.L,num_points)
        y = func(x,n)
        return torch.trapezoid(y)

    def f(self,x):
        """f(x) = u(x,0)"""
        return 2*torch.sin(x * torch.pi / self.L)
    
    def f_helper(self,x,n):
        return self.f(x) * torch.sin(n*torch.pi*x / self.L)

    def g(self,x):
        """g(x) = u_{t}(x,0)"""
        return 0

    def g_helper(self,x,n):
        return self.g(x) * torch.sin(n*torch.pi*x / self.L)


    def solution(self,x,t):
        N = 10
        A = torch.tensor([(2/self.L) * self.integrate(self.f_helper,n,100) for n in range(1,N+1)])
        B = torch.tensor([(2/self.L) * self.integrate(self.g_helper,n,100) for n in range(1,N+1)])
        sin_t = torch.tensor([[self.L * torch.sin(n*self.v*torch.pi*t/(self.L)) / (n*torch.pi*self.v)] for n in range(1,N+1)])
        cos_t = torch.tensor([torch.cos(n*self.v*torch.pi*t/(self.L)) for n in range(1,N+1)])
        sin_x = torch.tensor([torch.sin(n*torch.pi*x/(self.L)) for n in range(1,N+1)])
        return torch.sum((A * sin_t + B*cos_t) * sin_x)

    def trial(self,x,t,n_out):
        """A(x,t) + x(L-x)*t**2*n_out
            A(x,t) = f(x) - [((L-x)/L)f(0) + (x/L)f(L)] + t{g(x)-[((L-x)/L)g(0)+(x/L)g(L)]}
            A(0,t) = f(0) - [f(0)] + t {g(0) - [g(0)]} = 0
            A(L,t) = f(L) - f(L) + t {g(L) - g(L)} = 0
            A(x,0) = f(x) - [((L-x)/L)f(0) + (x/L)f(L)] = f(x) (since f(0) = f(L) = 0)
            A_{t}(x,0) = g(x)-[((L-x)/L)g(0)+(x/L)g(L)] = g(x) (since g(0) = g(L) = 0 )
        """
        # trial_term guarantees to satisfy boundary conditions
        return self.f(x) + t*self.g(x) + x*(self.L-x)*(t**2)*n_out
    
    def diffEq(self,x,t,trial):
        trial_dx = grad(trial, x, torch.ones_like(trial), create_graph=True)[0]
        trial_dx2 = grad(trial_dx, x, torch.ones_like(trial_dx), create_graph=True)[0]
        trial_dt = grad(trial, t, torch.ones_like(trial), create_graph=True)[0]
        trial_dt2 = grad(trial_dt, t, torch.ones_like(trial_dt),create_graph=True)[0]
        return trial_dt2 - self.v**2 * trial_dx2


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            n_out = network(batch).view(-1,1)
            x, t = batch[:,0].view(-1,1), batch[:,1].view(-1,1)

            # Get value of trial solution f(x,t)
            trial = diffEq.trial(x,t,n_out)
    
            # Get value of diff equations D(x,t) = 0
            D = diffEq.diffEq(x, t, trial)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))

            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            
        if epoch%(epochs/5)==0:
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, trange)
            
        #store final loss of each epoch
        cost_list.append(loss.detach().numpy())
    
    network.train(False)
    return cost_list

def plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, trange):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
    T  = torch.linspace(trange[0],trange[1],num_samples, requires_grad=True)
    x,t = torch.meshgrid(X,T)
    input = torch.cat((x.reshape(-1,1),t.reshape(-1,1)),1)
    N = network.forward(input)
    trial = diffEq.trial(x.reshape(-1,1),t.reshape(-1,1),N)
    trial = trial.reshape(num_samples,num_samples).detach().numpy()

    exact = diffEq.solution(x,t).detach().numpy()
    x = x.detach().numpy()
    t = t.detach().numpy()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x,t,trial, rstride=1, cstride=1,
                cmap='plasma', edgecolors = 'none')
    # surf._facecolors2d = surf._facecolor3d
    # surf._edgecolors2d = surf._edgecolor3d
    # plt.colorbar(surf, location = 'left')
    ax.scatter(x,t,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()

network     = Fitter(num_hidden_nodes=10)
loss_fn      = torch.nn.MSELoss()
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-2)

# j = 4
# ranges = []
# for i in range(j):
#     lst = [i/j, (i+1)/j]
#     ranges.append(lst)
# ranges.append([0,1])
ranges = [[0,1]]
for xrange in ranges:
    trange = xrange
    num_samples = 10
    diffEq = DiffEq(xrange, trange, 1,1)
    train_set    = DataSet(xrange,trange,num_samples)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True) 
 
    losses = [1]
    iterations = 0
    epochs = 5000
    while losses[-1] > 0.001  and iterations < 10:
        newLoss = train(network, train_loader, loss_fn,
                            optimiser, diffEq, epochs, iterations)
        losses.extend(newLoss)
        iterations += 1
    losses = losses[1:]
    print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()

plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])