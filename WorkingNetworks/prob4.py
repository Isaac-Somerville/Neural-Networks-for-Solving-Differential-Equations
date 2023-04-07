#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
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
    
class DESolver(torch.nn.Module):
    """
    The neural network object, with 1 node in the input layer,
    1 node in the output layer, and 1 hidden layer with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layer

        Returns:
        DESolver object (neural network) with two attributes:
        fc1 (fully connected layer) -- linear transformation of hidden layer
        fc2 (fully connected layer) -- linear transformation of outer layer
        """
        super(DESolver, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 1, out_features = numHiddenNodes)
        self.fc2 = torch.nn.Linear(in_features = numHiddenNodes, out_features = 2)

    def forward(self, x):
        """
        Function which connects inputs to outputs in the neural network.

        Arguments:
        x (PyTorch tensor shape (batchSize,1)) -- input of neural network

        Returns:
        y (PyTorch tensor shape (batchSize,2)) -- output of neural network
        """
        # sigmoid activation function used on hidden layer
        h = torch.tanh(self.fc1(x))
        # Linear activation function used on outer layer
        y = self.fc2(h)
        return y
    
def f1Trial(x, n1_out):
    """Trial solution f1(x) to first DE"""
    return x * n1_out

def df1Trial( x, n1_out, dn1dx):
    """Derivative of trial solution f1'(x) to first DE"""
    return n1_out + (x * dn1dx)

def f2Trial(x, n2_out):
    """Trial solution f2(x) to second DE"""
    return 1 + (x * n2_out)

def df2Trial(x, n2_out, dn2dx):
    """Derivative of trial solution f2'(x) to second DE"""
    return n2_out + (x * dn2dx)

def diffEq1(x, f1_trial, f2_trial, df1_trial):
    """Returns D1(x) where first DE is D1(x) = 0"""
    LHS = df1_trial
    RHS = torch.cos(x) + (f1_trial**2 + f2_trial) - (1 + x**2 + torch.sin(x)**2)
    return LHS - RHS
# - + , + - always works
# original is + -, - +
def diffEq2(x, f1_trial, f2_trial, df2_trial):
    """Returns D2(x) where second DE is D2(x) = 0"""
    LHS = df2_trial
    RHS = 2*x - ((1 + x**2)*torch.sin(x)) + (f1_trial*f2_trial)
    return LHS - RHS

def solution1(x):
    """Analytic solution to first DE"""
    return torch.sin(x)

def solution2(x):
    """Analytic solution to second DE"""
    return 1 + x**2

def train(network, loader, lossFn, optimiser, numEpochs):
    """
    A function to train a neural network to solve a pair of coupled 
    first-order ODEs with Dirichlet boundary conditions.

    Arguments:
    network (Module) -- the neural network
    loader (DataLoader) -- generates batches from the training dataset
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    numEpochs (int) -- number of training epochs

    Returns:
    cost_list (list of length 'numEpochs') -- cost values of all epochs
    """
    cost_list=[]
    network.train(True) # set module in training mode
    for epoch in range(numEpochs):
        for batch in loader:
            n_out = network.forward(batch)

            # Separate two columns of output (one for f1, one for f2)
            # Using torch.split retains tensor history for autograd
            n1_out, n2_out = torch.split(n_out, split_size_or_sections=1, dim=1)

            # Get the derivative of both networks' outputs with respect to the input values. 
            dn1dx = grad(n1_out, batch, torch.ones_like(n1_out), retain_graph=True, create_graph=True)[0]
            dn2dx = grad(n2_out, batch, torch.ones_like(n2_out), retain_graph=True, create_graph=True)[0]

            # Get value of trial solutions f1(x), f2(x)
            f1_trial = f1Trial(batch, n1_out)
            f2_trial = f2Trial(batch, n2_out)
            # Get f1'(x) and f2'(x)
            df1_trial = df1Trial(batch, n1_out, dn1dx)
            df2_trial = df2Trial(batch, n2_out, dn2dx)
            # Get LHS of differential equations D1(x) = 0, D2(x) = 0
            D1 = diffEq1(batch, f1_trial, f2_trial, df1_trial)
            D2 = diffEq2(batch, f1_trial, f2_trial, df2_trial)

            # Calculate and store cost
            cost1 = lossFn(D1, torch.zeros_like(D1))
            cost2 = lossFn(D2, torch.zeros_like(D2))
            cost = cost1 + cost2
            
            cost.backward() # perform backpropagation
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero

        cost_list.append(cost.detach().numpy()) # store cost of each epoch
        
    network.train(False) # set module out of training mode
    return cost_list


def plotNetwork(network, epoch):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    x    = torch.linspace(totalXRange[0], totalXRange[1], 36, requires_grad=True).view(-1,1)
    n_out    = network.forward(x)
    n1_out, n2_out = torch.split(n_out, split_size_or_sections=1, dim=1)

    # Get the derivative of both networks' outputs with respect to the input values. 
    dn1dx = grad(n1_out, x, torch.ones_like(n1_out), retain_graph=True, create_graph=True)[0]
    dn2dx = grad(n2_out, x, torch.ones_like(n2_out), retain_graph=True, create_graph=True)[0]

    # Get value of trial solutions f1(x), f2(x)
    f1_trial = f1Trial(x, n1_out)
    f2_trial = f2Trial(x, n2_out)
    # Get f1'(x) and f2'(x)
    df1_trial = df1Trial(x, n1_out, dn1dx)
    df2_trial = df2Trial(x, n2_out, dn2dx)
    # Get LHS of differential equations D1(x) = 0, D2(x) = 0
    D1 = diffEq1(x, f1_trial, f2_trial, df1_trial)
    D2 = diffEq2(x, f1_trial, f2_trial, df2_trial)

    # Calculate and store cost
    cost1 = lossFn(D1, torch.zeros_like(D1))
    cost2 = lossFn(D2, torch.zeros_like(D2))
    cost = cost1 + cost2
    print("test cost = ", cost.item())

    exact1 = solution1(x)
    exact2 = solution2(x)
    MSE1 = lossFn(f1_trial, exact1)
    MSE2 = lossFn(f2_trial, exact2)
    print("MSE between trial and exact solutions = ", ((MSE1 + MSE2)/2).item())

    x = x.detach().numpy()
    exact1 = exact1.detach().numpy()
    exact2 = exact2.detach().numpy()
    f1_trial = f1_trial.detach().numpy()
    f2_trial = f2_trial.detach().numpy()
    
    # plt.plot(x, f1_trial, 'r-', label = "f\u2081(x) Trial Solution")
    # plt.plot(x, exact1, 'b.', label = "f\u2081(x) Exact Solution")
    # plt.xlabel("x",fontsize = 16)
    # plt.ylabel("y",fontsize = 16)
    # plt.legend(loc = "lower center", fontsize = 16)
    # plt.title("Network 1: " + str(epoch) + " Epochs", fontsize = 16)
    # plt.show()
    
    # plt.plot(x, f2_trial, 'r-', label = "f\u2082(x) Trial Solution")
    # plt.plot(x, exact2, 'b.', label = "f\u2082(x) Exact Solution")
    
    # plt.xlabel("x",fontsize = 16)
    # plt.ylabel("y",fontsize = 16)
    # plt.legend(loc = "upper left",fontsize = 16)
    # plt.title("Network 2: " + str(epoch) + " Epochs",fontsize = 16)
    # plt.show()
        
    plt.plot(x, f1_trial, 'r-', label = "f\u2081(x) Trial Solution")
    plt.plot(x, exact1, 'b.', label = "f\u2081(x) Exact Solution")
    
    plt.plot(x, f2_trial, 'm-', label = "f\u2082(x) Trial Solution")
    plt.plot(x, exact2, 'c.', label = "f\u2082(x) Exact Solution")
    
    plt.xlabel("x",fontsize = 16)
    plt.ylabel("y",fontsize = 16)
    plt.legend(loc = "upper left",fontsize = 16)
    plt.title("Example 4: " + str(epoch) + " Epochs",fontsize = 16)
    plt.show()


# try: # load saved network if possible
#     checkpoint = torch.load('problem4InitialNetwork.pth')
#     network    = checkpoint['network']
# except: # create new network
#     network    = DESolver(numHiddenNodes=16)
#     checkpoint = {'network': network}
#     torch.save(checkpoint, 'problem4InitialNetwork.pth')
network     = DESolver(numHiddenNodes=16)
lossFn      = torch.nn.MSELoss()
optimiser   = torch.optim.Adam(network.parameters(), lr = 1e-3)
totalXRange      = [0,3]
numTotalSamples  = 30

###### TRAINING ON SUBINTERVALS OF INCREASING SIZE, STARTING FROM 0
# ranges = [[0,3]]  # fails 8.423211693298072e-05, 4.97
# ranges = [[0,1.5], [0,3]]  # succeeds 7.001220365054905e-05, 3.14 \times 10^{-4}
# ranges = [[0,1],[0,2],[0,3]]  # succeeds  1.481836170569295e-05 2.63 \times 10^{-4}

###### TRAINNG ON NON-INTERSECTING SUBERINTERVALS, STARTING FROM 0
# ranges = [[0,1.5], [1.5,3]] # succeeds but suboptimal solution 6.24 \times 10^{-3}, 1.85 \times 10^{-3}
# ranges = [[0,1],[1,2],[2,3]] # succeeds but suboptimal solution 5.30 \times 10^{-3}, 1.62 \times 10^{-3}

###### TRAINNG ON NON-INTERSECTING SUBERINTERVALS, THEN FULL INTERVAL, STARTING FROM 0
# ranges = [[0,1.5], [1.5,3], [0,3]] # succeeds 2.2057407477404922e-05  1.84 \times 10^{-4}
# ranges = [[0,1],[1,2],[2,3],[0,3]] # succeeds 7.5498546721064486e-06, 1.69 \times 10^{-4}

###### TRAINING ON SUBINTERVALS OF INCREASING SIZE, STARTING FROM 3
# ranges = [[1.5,3], [0,3]]  # fails 1.93 \times 10^{-4}, 5.80
# ranges = [[2,3],[1,3],[0,3]]  # fails, 2.58 \times 10^{-4}, 6.35
# ranges = [[2.25,3],[1.5,3],[0.75,3],[0,3]] # fails 7.86 \times 10^{-4}, 7.75

###### TRAINNG ON NON-INTERSECTING SUBERINTERVALS, THEN FULL INTERVAL, STARTING FROM 3
# ranges = [[1.5,3], [0,1.5], [0,3]] # succeeds 6.55 \times 10^{-4}, 4.39 \times 10^{-4}
# ranges = [[2,3],[1,2],[0,1],[0,3]] # succeeds 1.82 \times 10^{-5}, 7.85 \times 10^{-4}

###### TRAINNG ON NON-INTERSECTING SUBERINTERVALS, STARTING FROM 3
# ranges = [[1.5,3], [0,1.5]] # succeeds, 7.21 \times 10^{-3} 1.62 \times 10^{-3}
ranges = [[2,3],[1,2],[0,1]] # fails

costList = []
epoch = 0
numEpochs = 1000
totalEpochs = 36000
epochsPerSubRange = int(totalEpochs / len(ranges))
for subRange in ranges:
    epochCounter = 0
    numSamples = int(10 * (subRange[1] - subRange[0]))
    trainData    = DataSet(numSamples, subRange)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples), shuffle=True)

    while epochCounter < epochsPerSubRange:
        costList.extend(train(network, trainLoader, lossFn, optimiser, numEpochs))
        epoch += numEpochs
        epochCounter += numEpochs
        # if epochCounter % 10000 == 0:
        #     plotNetwork(network,epoch)
    
    plt.semilogy(costList)
    plt.xlabel("Epochs",fontsize = 16)
    plt.ylabel("Cost",fontsize = 16)
    plt.title("Example 4: Training Cost",fontsize = 16)
    plt.show()
    plotNetwork(network,epoch)

    
print(f"{totalEpochs} epochs total, final cost = {costList[-1]}")
# plt.semilogy(costList)
# plt.xlabel("numEpochs", fontsize = 16)
# plt.ylabel("Error", fontsize = 16)
# plt.title("Error", fontsize = 16)
# plt.show()
# plotNetwork(network, epoch)
# %%

# %%

