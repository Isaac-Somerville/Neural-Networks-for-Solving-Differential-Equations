#%%

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x-coordinates as test data"""
    def __init__(self, num_samples, xrange):
        self.data_in  = torch.linspace(xrange[0], xrange[1], num_samples, requires_grad=True)

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
    
class DiffEq1():
    def __init__(self, xrange, num_samples):
        self.xrange = xrange
        self.num_samples = num_samples
        
    def solution1(x):
        return torch.sin(x)
    
    def f1_trial(x,n1_out):
        return x * n1_out
    
    def df1_trial(x,n1_out,dn1dx):
        return n1_out + x*dn1dx
    
    def solution2(x):
        return 1 + x**2
    
    def f2_trial(x, n2_out):
        return 1 + x*n2_out
    
    def df2_trial(x, n2_out, dn2dx):
        return n2_out + x * dn2dx
    
    #TODO: questa roba qui
    def diffEq1(x, f1_trial, f2_trial, dn1dx):
        return
    
    def diffEq2(x, f1_trial, f2_trial, dn2dx):
        return

def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            x = batch.view(-1, 1)
            n_out = network(x)

            # Get the derivative of the network output with respect
            # to the input values. 
            dndx = torch.autograd.grad(n_out, x, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            d2ndx2 = torch.autograd.grad(dndx, x, torch.ones_like(dndx), retain_graph=True, create_graph=True)[0]
            
            # Get value of trial solution f(x)
            f_trial = trialFunc(x, n_out)
            # Get df / dx
            df_trial = dTrialFunc(x, n_out, dndx)
            # Get d^2f / dx^2
            d2f_trial = d2TrialFunc(x,n_out,dndx,d2ndx2)
            # Get LHS of diff equation D(x) = 0
            diff_eq = diffEq(x, f_trial, df_trial, d2f_trial)
            
            loss = loss_fn(diff_eq, torch.zeros_like(diff_eq))
            cost_list.append(loss.detach().numpy())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        if epoch%(epochs/5)==0:
            plotNetwork(network, solution,trialFunc, epoch, epochs, iterations)
    network.train(False)
    return cost_list


def plotNetwork(network, solution, trialFunc, epoch, epochs, iterations):
    x    = torch.Tensor(np.linspace(xrange[0], xrange[1], num_samples)).view(-1,1)
    N    = network.forward(x).detach().numpy()
    exact = solution(x).detach().numpy()
    plt.plot(x, trialFunc(x,N), 'r-', label = "Neural Network Output")
    plt.plot(x, exact, 'b.', label = "True Solution")
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc = "lower right")
    plt.title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    
    
def solution3(x):
    """
    Analytic solution to Lagaris problem 3
    """
    return torch.exp(-x/5) * torch.sin(x)

def trialFunc3(x, n_out):
    """
    Trial solution to Lagaris problem 3
    f(x) = x + x^2 * N(x)
    """ 
    return x + (x**2 * n_out)

def dTrialFunc3(x, n_out, dndx):
    """
    Derivative of trial solution to Lagaris problem 3
    f'(x) = 1 + 2xN(x) + x^2 * N'(x)
    """ 
    return 1 + (2*x*n_out) + (x**2 * dndx)

def d2TrialFunc3(x,n_out,dndx,d2ndx2):
    """
    Second derivative of trial solution to Lagaris problem 3
    f''(x) = 2N(x) + (4x * N'(x)) + x^2 N''(x)
    """ 
    return 2*n_out + (4*x*dndx) + (x**2 * d2ndx2)

def diffEq3(x, f_trial, df_trial, d2f_trial):
    """
    Returns LHS of differential equation D(x) = 0
    for Lagaris problem 1
    """
    LHS = d2f_trial + (1/5)*df_trial + f_trial
    RHS = -(1/5) * torch.exp(-x/5) * torch.cos(x)
    return LHS - RHS
    
xrange=[0, 2]
num_samples = 30
network      = Fitter(num_hidden_nodes=10)
train_set    = DataSet(num_samples,  xrange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=60, shuffle=True)
loss_fn      = torch.nn.MSELoss()
optimiser    = torch.optim.Adam(network.parameters(), lr=1e-2)

solution = solution3
trialFunc = trialFunc3
dTrialFunc = dTrialFunc3
d2TrialFunc = d2TrialFunc3
diffEq = diffEq3

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.001 and iterations < 10:
    losses.extend( train(network, train_loader, loss_fn, optimiser, solution, 
                         trialFunc, dTrialFunc, d2TrialFunc, diffEq, epochs, iterations))
    iterations += 1
losses = losses[1:]
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")