#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, xrange, yrange, num_samples):
        X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
        Y  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
        # create tuple of (num_samples x num_samples) points
        x,y = torch.meshgrid(X,Y) 

        # input of forward function must have shape (batch_size, 2)
        self.data_in = torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)

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
    Differential equation from Lagaris et al. problem 5
    This problem is a PDE in two variables, with Dirichlet
    boundary conditions
    """
    def __init__(self, xrange, yrange, num_samples):
        self.xrange = xrange
        self.yrange = yrange
        self.num_samples = num_samples

    def solution(self,x,y):
        return torch.exp(-x) * (x + y**3)

    def trial(self,x,y,n_out):
        # trial_term guarantees to satisfy boundary conditions
        e_inv = torch.exp(-torch.ones_like(x))
        trial_term = (((1-x)*(y**3)) + (x*(1+(y**3))*e_inv) + ((1-y)*x*(torch.exp(-x)-e_inv)) +
                (y * (((1+x)*torch.exp(-x)) - (1-x+(2*x*e_inv)))))
        return (trial_term + x*(1-x)*y*(1-y)*n_out)
    
    def diffEq(self,x,y,trial):
        trial_dx = grad(trial, x, torch.ones_like(trial), create_graph=True)[0]
        trial_dx2 = grad(trial_dx, x, torch.ones_like(trial_dx), create_graph=True)[0]
        trial_dy = grad(trial, y, torch.ones_like(trial), create_graph=True)[0]
        trial_dy2 = grad(trial_dy, y, torch.ones_like(trial_dy),create_graph=True)[0]
        RHS = torch.exp(-x) * (x - 2 + y**3 + 6*y)
        return trial_dx2 + trial_dy2 - RHS


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            n_out = network(batch).view(-1,1)
            x, y = batch[:,0].view(-1,1), batch[:,1].view(-1,1)

            # Get value of trial solution f(x,y)
            trial = diffEq.trial(x,y,n_out)
    
            # Get value of diff equations D(x,y) = 0
            D = diffEq.diffEq(x, y, trial)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))

            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            
        # if epoch%(epochs/5)==0:
        if epoch == epochs:
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange)
            
        #store final loss of each epoch
        cost_list.append(loss.detach().numpy())
    
    network.train(False)
    return cost_list

def plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
    Y  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
    x,y = torch.meshgrid(X,Y)
    input = torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)
    N = network.forward(input)

    trial = diffEq.trial(x.reshape(-1,1),y.reshape(-1,1),N)
    trial = trial.reshape(num_samples,num_samples).detach().numpy()
    exact = diffEq.solution(x,y).detach().numpy()
    surfaceLoss = ((trial-exact)**2).mean()
    
    x = x.detach().numpy()
    y = y.detach().numpy()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(x,y,trial, rstride=1, cstride=1,
                cmap='plasma', edgecolors = 'none')
    # surf._facecolors2d = surf._facecolor3d
    # surf._edgecolors2d = surf._edgecolor3d
    # plt.colorbar(surf, location = 'left')
    ax.scatter(x,y,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    return surfaceLoss

# network     = Fitter(num_hidden_nodes=10)
# loss_fn      = torch.nn.MSELoss()
#optimiser  = torch.optim.Adam(network.parameters(), lr = 5e-3)

# check final loss for different lrs, fixed no. of epochs

# j = 4
# ranges = []
# for i in range(j):
#     lst = [i/j, (i+1)/j]
#     ranges.append(lst)
# ranges.append([0,1])
ranges = [[0,1]]
lrs = [(1e-2 + i * 4e-4) for i in range(10)]
# lrs = [(5e-3 + i * 1e-3) for i in range(1,11)]
# lrs = [(i * 1e-3) for i in range(1,11)]
finalLosses = []
surfaceLosses = []
for xrange in ranges:
    for lr in lrs:
        network     = Fitter(num_hidden_nodes=8)
        loss_fn      = torch.nn.MSELoss()
        optimiser  = torch.optim.Adam(network.parameters(), lr = lr)
        yrange = xrange
        num_samples = 8
        diffEq = DiffEq(xrange, yrange, num_samples)
        train_set    = DataSet(xrange,yrange,num_samples)
        train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=32, shuffle=True) 
    
        losses = [1]
        iterations = 0
        epochs = 5000
        while losses[-1] > 0.001  and iterations < 1:
            newLoss = train(network, train_loader, loss_fn,
                                optimiser, diffEq, epochs, iterations)
            losses.extend(newLoss)
            iterations += 1
        losses = losses[1:]
        finalLoss = losses[-1]
        finalLosses.append(finalLoss)
        print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

        plt.semilogy(losses)
        plt.xlabel("Epochs")
        plt.ylabel("Log of Loss")
        plt.title("Loss")
        plt.show()

        surfaceLoss = plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])
        surfaceLosses.append(surfaceLoss)

plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])
plt.show()
plt.semilogy(lrs,surfaceLosses)
plt.xlabel("Learning Rate")
plt.ylabel("Mean Squared Error ")
plt.title("Mean Squared Error of Network from Exact Solution")


#%%