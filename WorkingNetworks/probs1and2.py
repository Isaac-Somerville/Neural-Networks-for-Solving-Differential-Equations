#%%

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x-coordinates as test data"""
    def __init__(self, numSamples, xrange):
        self.dataIn  = torch.linspace(xrange[0], xrange[1], numSamples, requires_grad=True).view(-1,1)

    def __len__(self):
        return len(self.dataIn)

    def __getitem__(self, i):
        return self.dataIn[i]
    
class Fitter(torch.nn.Module):
    """
    The neural network object, with 1 node in the input layer,
    1 node in the output layer, and 1 hidden layer with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layer

        Returns:
        Fitter object (neural network) with two attributes:
        fc1 (fully connected layer) -- linear transformation of hidden layer
        fc2 (fully connected layer) -- linear transformation of outer layer
        """
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 1, out_features = numHiddenNodes)
        self.fc2 = torch.nn.Linear(in_features = numHiddenNodes, out_features = 1)

    def forward(self, x):
        """
        Function which connects inputs to outputs in the neural network.

        Arguments:
        x (PyTorch tensor shape (batchSize,1)) -- input of neural network

        Returns:
        y (PyTorch tensor shape (batchSize,1)) -- output of neural network
        """
        # tanh activation function used on hidden layer
        h = torch.tanh(self.fc1(x))
        # Linear activation function used on outer layer
        y = self.fc2(h)
        return y
    
def train(network, loader, lossFn, optimiser, numEpochs):
    """Trains the neural network"""
    cost_list=[]
    network.train(True) # set module in training mode
    for epoch in range(numEpochs):
        for batch in loader:
            n_out = network.forward(batch) # network output

            # Derivative of the network output w.r.t. the input values:
            dndx = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True)[0]
            # Get value of trial solution f(x)
            f_trial = trialFunc(batch, n_out)
            # Get df / dx
            df_trial = dTrialFunc(batch, n_out, dndx)
            # Get LHS of differential equation G(x) = 0
            diff_eq = diffEq(batch, f_trial, df_trial)
            
            cost = lossFn(diff_eq, torch.zeros_like(diff_eq)) # calculate cost
            cost.backward() # perform backpropagation
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero

        cost_list.append(cost.detach().numpy())# store cost of each epoch
    network.train(False) # set module out of training mode
    return cost_list


def plotNetwork(network, epoch):
    '''
    Plots the output of the neural network and the analytic solution
    '''
    x    = torch.linspace(xrange[0], xrange[1], 50).view(-1,1)
    x.requires_grad = True
    N    = network.forward(x)
    f_trial = trialFunc(x, N)
    dndx = grad(N, x, torch.ones_like(N), retain_graph=True, create_graph=True)[0]
    df_trial = dTrialFunc(x, N, dndx)
    diff_eq = diffEq(x, f_trial, df_trial)
    cost = lossFn(diff_eq, torch.zeros_like(diff_eq))
    print("test cost = ", cost.item())
    
    exact = solution(x).detach().numpy()
    x = x.detach().numpy()
    N = N.detach().numpy()
    plt.plot(x, trialFunc(x,N), 'r-', label = "Neural Network Output")
    plt.plot(x, exact, 'b.', label = "True Solution")
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc = "upper left")
    plt.title(str(epoch) + " Epochs")
    plt.show()
    
def solution1(x):
    """
    Analytic solution to Lagaris problem 1
    """
    return (torch.exp(-(x**2)/2) / (1 + x + x**3))+ x**2 

def trialFunc1(x, n_out):
    """
    Trial solution to Lagaris problem 1
    f(x) = 1 + xN(x)
    """ 
    return 1 + x * n_out

def dTrialFunc1(x, n_out, dndx):
    """
    Derivative of trial solution to Lagaris problem 1
    f'(x) = N(x) + xN'(x)
    """ 
    return n_out + x * dndx

def diffEq1(x, f_trial, df_trial):
    """
    Returns LHS of differential equation G(x) = 0
    for Lagaris problem 1
    """
    RHS = x**3 + 2*x + (x**2 * ((1+3*x**2) / (1 + x + x**3)))
    LHS = df_trial + ((x + (1+3*(x**2)) / (1 + x + x**3) ) * f_trial)
    return LHS - RHS
    
def solution2(x):
    """
    Analytic solution to Lagaris problem 2
    """
    return torch.exp(-x/5) * torch.sin(x)

def trialFunc2(x, n_out):
    """
    Trial solution to Lagaris problem 2
    f(x) = x * N(x)
    """ 
    return x * n_out

def dTrialFunc2(x, n_out, dndx):
    """
    Derivative of trial solution to Lagaris problem 2
    f'(x) = N(x) + x * N'(x)
    """ 
    return n_out + x * dndx

def diffEq2(x, f_trial, df_trial):
    """
    Returns LHS of differential equation G(x) = 0
    for Lagaris problem 2
    """
    RHS = df_trial + (1/5)*f_trial
    LHS = torch.exp(-x/5) * torch.cos(x)
    return LHS - RHS
    
xrange=[0, 2]
numSamples = 10
network      = Fitter(numHiddenNodes=10)
train_set    = DataSet(numSamples,  xrange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=10, shuffle=True)
lossFn      = torch.nn.MSELoss()
optimiser    = torch.optim.SGD(network.parameters(), lr=1e-3)

solution = solution1
trialFunc = trialFunc1
dTrialFunc = dTrialFunc1
diffEq = diffEq1

# solution = solution2
# trialFunc = trialFunc2
# dTrialFunc = dTrialFunc2
# diffEq = diffEq2

costList = []
epoch = 0
numEpochs = 1000
start = time.time()
while epoch < 1000:
    costList.extend(train(network, train_loader, lossFn, optimiser, numEpochs))
    epoch += numEpochs
    plotNetwork(network, epoch)
end = time.time()
print(epoch, "epochs total, final cost = ", costList[-1])
print("total time elapsed = ", end - start, " seconds")

plt.semilogy(costList)
plt.xlabel("Epochs",fontsize = 16)
plt.ylabel("Cost",fontsize = 16)
plt.title("Network Training Cost",fontsize = 16)

# %%
