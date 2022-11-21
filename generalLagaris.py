#%%
"""
HW: read Nielsen
Look into local minima stuff
Generalise this code
"""

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

def train(network, loader, loss_fn, optimiser, solution, trialFunc, dTrialFunc, diffEq, epochs, iterations):
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
            
            # Get value of trial solution f(x)
            f_trial = trialFunc(x, n_out)
            # Get df / dx
            df_trial = dTrialFunc(x, n_out, dndx)
            # Get LHS of diff equation D(x) = 0
            diff_eq = diffEq(x, f_trial, df_trial)
            
            loss = loss_fn(diff_eq, torch.zeros_like(diff_eq))
            cost_list.append(loss.detach().numpy())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        if epoch%(epochs/5)==0:
            plotNetwork(network, solution, epoch, epochs, iterations)
    network.train(False)
    return cost_list


def plotNetwork(network, solution, epoch, epochs, iterations):
    x    = torch.Tensor(np.linspace(xrange[0], xrange[1], num_samples)).view(-1,1)
    N    = network.forward(x).detach().numpy()
    exact = solution(x).detach().numpy()
    plt.plot(x, x*N, 'r-', label = "Neural Network Output")
    plt.plot(x, exact, 'b.', label = "True Solution")
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc = "upper right")
    plt.title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    
def solution1(x):
    """
    Analytic solution to Lagaris problem 1
    """
    return (torch.exp(-(x**2)/2) / (1 + x + x**3)) + x**2 

def trialFunc1(x, n_out):
    """
    Trial solution to Lagaris problem 1
    f(x) = 1 + x * N(x)
    """ 
    return 1 + x * n_out

def dTrialFunc1(x, n_out, dndx):
    """
    Derivative of trial solution to Lagaris problem 1
    f'(x) = N(x) + x * N'(x)
    """ 
    return n_out + x * dndx

def diffEq1(x, f_trial, df_trial):
    """
    Returns LHS of differential equation D(x) = 0
    for Lagaris problem 1
    """
    RHS = x**3 + 2*x + (x**2 * (1+3*x**2) / (1 + x + x**3))
    LHS = df_trial + (x + (1+3*x**2) / (1 + x + x**3) ) * f_trial
    return LHS - RHS
    
xrange=[0, 10]
num_samples = 30
network      = Fitter(num_hidden_nodes=10)
train_set    = DataSet(num_samples,  xrange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=60, shuffle=True)
loss_fn      = torch.nn.MSELoss()
optimiser    = torch.optim.Adam(network.parameters(), lr=1e-3)

solution = solution1
trialFunc = trialFunc1
dTrialFunc = dTrialFunc1
diffEq = diffEq1

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.01 and iterations < 10:
    losses.extend( train(network, train_loader, loss_fn, optimiser, solution, 
                         trialFunc, dTrialFunc, diffEq, epochs, iterations))
    iterations += 1
losses = losses[1:]
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")

# %%
