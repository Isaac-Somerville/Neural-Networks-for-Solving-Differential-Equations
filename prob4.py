#%%

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x-coordinates as test data"""
    def __init__(self, num_samples, xrange):
        self.data_in  = torch.linspace(xrange[0], xrange[1], num_samples, requires_grad=True).view(-1,1)

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, i):
        return self.data_in[i]
    
class Fitter(torch.nn.Module):
    """Forward propagations"""
    def __init__(self, num_hidden_nodes):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, x):
        hidden = torch.sigmoid(self.fc1(x))
        y = self.fc2(hidden)
        return y
    
class DiffEq():
    """
    Differential equations from Lagaris et al. problem 4
    This problem has a system of two coupled differential equations
    D1(x) = 0 and D2(x) = 0
    """
    def __init__(self, xrange, num_samples):
        self.xrange = xrange
        self.num_samples = num_samples
        
    def solution1(self, x):
        """Analytic solution to first DE"""
        return torch.sin(x)
    
    def f1_trial(self, x, n1_out):
        """Trial solution f1(x) to first DE"""
        return x * n1_out
    
    def df1_trial(self, x, n1_out, dn1dx):
        """Derivative of trial solution f1'(x) to first DE"""
        return n1_out + x*dn1dx
    
    def solution2(self, x):
        """Analytic solution to second DE"""
        return 1 + x**2
    
    def f2_trial(self, x, n2_out):
        """Trial solution f2(x) to second DE"""
        return 1 + x*n2_out
    
    def df2_trial(self, x, n2_out, dn2dx):
        """Derivative of trial solution f2'(x) to second DE"""
        return n2_out + x * dn2dx
    
    def diffEq1(self, x, f1_trial, f2_trial, df1_trial):
        """Returns D1(x) where first DE is D1(x) = 0"""
        LHS = df1_trial
        RHS = torch.cos(x) + f1_trial**2 + f2_trial - (1 + x**2 + torch.sin(x)**2)
        return LHS - RHS
    
    def diffEq2(self, x, f1_trial, f2_trial, df2_trial):
        """Returns D2(x) where second DE is D2(x) = 0"""
        LHS = df2_trial
        RHS = 2*x - ((1+x**2)*torch.sin(x)) + (f1_trial*f2_trial)
        return LHS - RHS

def train(network1, network2, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network1.train(True)
    network2.train(True)
    for epoch in range(epochs+1):
        for x in loader:
            n1_out = network1(x)
            n2_out = network2(x)

            # Get the derivative of both networks' outputs with respect
            # to the input values. 
            dn1dx = torch.autograd.grad(n1_out, x, torch.ones_like(n1_out), retain_graph=True, create_graph=True)[0]
            dn2dx = torch.autograd.grad(n2_out, x, torch.ones_like(n2_out), retain_graph=True, create_graph=True)[0]

            # Get value of trial solutions f1(x), f2(x)
            f1_trial = diffEq.f1_trial(x, n1_out)
            f2_trial = diffEq.f2_trial(x, n2_out)
            # Get df1 / dx and df2 / dx
            df1_trial = diffEq.df1_trial(x, n1_out, dn1dx)
            df2_trial = diffEq.df2_trial(x, n2_out, dn2dx)
            # Get LHS of diff equations D1(x) = 0, D2(x) = 0
            D1 = diffEq.diffEq1(x, f1_trial, f2_trial, df1_trial)
            D2 = diffEq.diffEq2(x, f1_trial, f2_trial, df2_trial)

            
            # Calculate and store loss
            loss1 = loss_fn(D1, torch.zeros_like(D1))
            loss2 = loss_fn(D2, torch.zeros_like(D2))
            loss = loss1 + loss2
            cost_list.append(loss.detach().numpy())
            
            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            
        if epoch%(epochs/5)==0:
            plotNetwork(network1, network2, diffEq, epoch, epochs, iterations)
        
    network1.train(False)
    network2.train(False)
    return cost_list


def plotNetwork(network1, network2, diffEq, epoch, epochs, iterations):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    x    = torch.Tensor(np.linspace(xrange[0], xrange[1], num_samples)).view(-1,1)
    N1   = network1.forward(x).detach().numpy()
    N2   = network2.forward(x).detach().numpy()
    exact1 = diffEq.solution1(x).detach().numpy()
    exact2 = diffEq.solution2(x).detach().numpy()
    
    plt.plot(x, diffEq.f1_trial(x,N1), 'r-', label = "Neural Network 1 Output")
    plt.plot(x, exact1, 'b.', label = "True Solution")
    
    plt.xlabel("x")
    plt.ylabel("f1(x)")
    plt.legend(loc = "lower center")
    plt.title("Network 1: " + str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    
    plt.plot(x, diffEq.f2_trial(x,N2), 'r-', label = "Neural Network 2 Output")
    plt.plot(x, exact2, 'b.', label = "True Solution")
    
    plt.xlabel("x")
    plt.ylabel("f2(x)")
    plt.legend(loc = "upper left")
    plt.title("Network 2: " + str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    
xrange=[0, 3]
num_samples = 30
diffEq = DiffEq(xrange, num_samples)
network1     = Fitter(num_hidden_nodes=20)
network2     = Fitter(num_hidden_nodes=20)
train_set    = DataSet(num_samples,  xrange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=60, shuffle=True)
loss_fn      = torch.nn.MSELoss()
params = list(network1.parameters()) + list(network2.parameters())
optimiser  = torch.optim.Adam(params, lr = 1e-2)

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.001  and iterations < 10:
    newLoss = train(network1, network2, train_loader, loss_fn,
                        optimiser, diffEq, epochs, iterations)
    losses.extend(newLoss)
    iterations += 1
losses = losses[1:]
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")
# %%
