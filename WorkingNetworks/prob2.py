#%%

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import grad

class DataSet(torch.utils.data.Dataset):
    """
    An object which generates the x values for the input node 
    """
    def __init__(self, numSamples, xRange):
        """
        Arguments:
        xRange (list of length 2) -- lower and upper limits for input values x
        numSamples (int) -- number of training data samples

        Returns:
        DataSet object with one attributes:
            dataIn (PyTorch tensor of shape (numSamples,1)) -- 'numSamples'
                evenly-spaced data points from xRange[0] to xRange[1]
        """
        self.dataIn  = torch.linspace(xRange[0], xRange[1], numSamples, requires_grad=True).view(-1,1)
        # 'view' method reshapes tensors, in this case into a column vector

    def __len__(self):
        """
        Arguments:
        None

        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return len(self.dataIn)

    def __getitem__(self, idx):
        """
        Used by DataLoader object to retrieve training data points

        Arguments:
        idx (int) -- index of data point required
        
        Returns:
        x (tensor shape (1,1)) -- data point at index 'idx'
        """
        return self.dataIn[idx]
    
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


def plotNetwork(network, algorithm, epoch):
    '''
    Plots the output of the neural network and the analytic solution
    '''
    x    = torch.linspace(xRange[0], xRange[1], 60, requires_grad=True).view(-1,1)
    N    = network.forward(x)
    f_trial = trialFunc(x, N)
    dndx = grad(N, x, torch.ones_like(N), retain_graph=True, create_graph=True)[0]
    df_trial = dTrialFunc(x, N, dndx)
    diff_eq = diffEq(x, f_trial, df_trial)
    cost = lossFn(diff_eq, torch.zeros_like(diff_eq))
    print("test cost = ", cost.item())
    
    exact = solution(x)
    MSECost = lossFn(f_trial, exact)
    print("MSE between trial and exact solution = ", MSECost.item())
    exact = exact.detach().numpy()
    x = x.detach().numpy()
    N = N.detach().numpy()
    plt.plot(x, trialFunc(x,N), 'r-', label = "Neural Network Output")
    plt.plot(x, exact, 'b.', label = "True Solution")
    
    plt.xlabel("x", fontsize = 16)
    plt.ylabel("f(x)", fontsize = 16)
    plt.legend(loc = "upper right", fontsize = 16)
    plt.title(str(algorithm) + ": " + str(epoch) + " Epochs", fontsize = 16)
    plt.show()
    return

def trialFunc2(x, n_out):
    """
    Trial solution to Lagaris problem 2: f(x) = x * N(x)
    Arguments:
        x (tensor of shape (batchSize,1)) -- input of neural network
        n_out (tensor of shape (batchSize,1)) -- output of neural network
    Returns:
        x * n_out (tensor of shape (batchSize,1)) -- trial solution to differential equation
    """ 
    return x * n_out

def dTrialFunc2(x, n_out, dndx):
    """
    Derivative of trial solution to Lagaris problem 2: f'(x) = N(x) + x * N'(x)
    Arguments:
        x (tensor of shape (batchSize,1)) -- input of neural network
        n_out (tensor of shape (batchSize,1)) -- output of neural network
        dndx (tensor of shape (batchSize,1)) -- derivative of n_out w.r.t. x
    Returns:
        n_out + x * dndx (tensor of shape (batchSize,1)) -- derivative of trial function w.r.t. x
    """ 
    return n_out + x * dndx

def diffEq2(x, f_trial, df_trial):
    """
    Returns D(x) of differential equation D(x) = 0 from Lagaris problem 2
    Arguments:
        x (tensor of shape (batchSize,1)) -- input of neural network
        f_trial (tensor of shape (batchSize,1)) -- trial solution at x
        df_trial (tensor of shape (batchSize,1)) -- derivative of trial solution at x
    Returns:
        LHS - RHS (tensor of shape (batchSize,1)) -- differential equation evaluated at x"""
    RHS = df_trial + (1/5)*f_trial
    LHS = torch.exp(-x/5) * torch.cos(x)
    return LHS - RHS

def solution2(x):
    """
    Analytic solution to Lagaris problem 1
    Arguments:
        x (tensor of shape (batchSize,1)) -- input of neural network
    Returns:
        y (tensor of shape (batchSize,1)) -- analytic solution of differential equation at x"""
    y = torch.exp(-x/5) * torch.sin(x)
    return y


def train(network, loader, lossFn, optimiser, numEpochs):
    """
    A function to train a neural network to solve a 
    first-order ODE with Dirichlet boundary conditions.

    Arguments:
    network (Module) -- the neural network
    loader (DataLoader) -- generates batches from the training dataset
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    numEpochs (int) -- number of training epochs

    Returns:
    costList (list of length 'numEpochs') -- cost values of all epochs
    """
    cost_list=[]
    network.train(True) # set module in training mode
    for epoch in range(numEpochs):
        for batch in loader:
            n_out = network.forward(batch) # network output

            # Get derivative of the network output w.r.t. the input values:
            dndx = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True)[0]
            # torch.ones_like(x) creates a tensor the same shape as x, filled with 1's
            
            # Get value of trial solution f(x)
            f_trial = trialFunc(batch, n_out)
            # Get df / dx
            df_trial = dTrialFunc(batch, n_out, dndx)
            # Get LHS of differential equation D(x) = 0
            diff_eq = diffEq(batch, f_trial, df_trial)
            
            cost = lossFn(diff_eq, torch.zeros_like(diff_eq)) # calculate cost
            # torch.zeros_like(x) creates a tensor the same shape as x, filled with 0's
            cost.backward() # perform backpropagation
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero

        cost_list.append(cost.detach().numpy())# store cost of each epoch
    network.train(False) # set module out of training mode
    return cost_list


solution    = solution2
trialFunc   = trialFunc2
dTrialFunc  = dTrialFunc2
diffEq      = diffEq2

try: # load saved network (initial state) and dictionary containing cost lists, if possible
    checkpoint = torch.load('problem2.pth')
    network    = checkpoint['network']
    costsDict  = torch.load('problem2Costs.pth')
except: # create new network and new dictionary to store cost lists
    network    = Fitter(numHiddenNodes=10)
    checkpoint = {'network': network}
    torch.save(checkpoint, 'problem2.pth')
    costsDict  = {}
xRange       = [0, 10]
numSamples   = 50
batchSize    = 50
train_set    = DataSet(numSamples, xRange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batchSize, shuffle=True)
lossFn       = torch.nn.MSELoss()

# algorithm    = "Batch Gradient Descent"
# optimiser    = torch.optim.SGD(network.parameters(), lr=1e-3)

# algorithm    = "Gradient Descent with Momentum"
# optimiser    = torch.optim.SGD(network.parameters(), lr=1e-3, momentum = 0.9, dampening = 0.9)

algorithm    = "RProp"
optimiser    = torch.optim.Rprop(network.parameters(), lr=1e-3)

# algorithm    = "RMSProp"
# optimiser    = torch.optim.RMSprop(network.parameters(), lr=1e-3)

# algorithm    = "Adam"
# optimiser    = torch.optim.Adam(network.parameters(), lr=1e-3)

costList    = []
epoch       = 0
numEpochs   = 1000
totalEpochs = 20000

start = time.time()
while epoch < totalEpochs:
    costList.extend(train(network, train_loader, lossFn, optimiser, numEpochs))
    epoch += numEpochs
end = time.time()

costsDict[algorithm] = costList # store cost list for each algorithm in a dictionary
torch.save(costsDict, 'problem2Costs.pth') # save dictionary

plotNetwork(network, algorithm, epoch)
plt.semilogy(costList)
plt.xlabel("Epochs",fontsize = 16)
plt.ylabel("Cost",fontsize = 16)
plt.title(str(algorithm) + ": " + "Training Cost",fontsize = 16)
plt.show()

print(epoch, "epochs total, final cost = ", costList[-1])
print("total time elapsed = ", end - start, " seconds")

# print all cost lists on same graph
costsDict = torch.load('problem2Costs.pth')
plt.plot(costsDict["Batch Gradient Descent"], label = "Batch GD")
plt.plot(costsDict["Gradient Descent with Momentum"], label = "GD with Momentum")
plt.plot(costsDict["RProp"], label = "RProp")
plt.plot(costsDict["RMSProp"], label = "RMSProp")
plt.plot(costsDict["Adam"], label = "Adam")
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Cost", fontsize = 16)
plt.legend(loc = "upper right", fontsize = 14)
plt.title("Optimisation Algorithm Cost Values", fontsize = 16)
# set x and y limits to improve visibility
ax = plt.gca()
ax.set_ylim([0, 0.05])
ax.set_xlim([100, 20100])
plt.show()

# %%
