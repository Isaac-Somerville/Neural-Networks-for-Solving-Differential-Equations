#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# TODO: CLEAN UP CODE 

class LinearDataSet(torch.utils.data.Dataset):
    """
    An object which generates a lattice of (x,y) values for the input node 
    """
    def __init__(self, xRange, yRange, numSamples):
        """
        Arguments:
        xRange (list of length 2) -- lower and upper limits for input values x
        yRange (list of length 2) -- lower and upper limits for input values y
        numSamples (int) -- number of training data samples along each axis

        Returns:
        DataSet object with one attributes:
            dataIn (PyTorch tensor of shape (numSamples^2,2)) -- 'numSamples'^2
                evenly-spaced grid points from (xRange[0], yRange[0]) to (xRange[1], yRange[1])
        """
        X  = torch.linspace(xRange[0],xRange[1],numSamples, requires_grad=True)
        Y  = torch.linspace(yRange[0],yRange[1],numSamples, requires_grad=True)
        grid = torch.meshgrid(X,Y)
        # meshgrid takes Cartesian product of tensors X and Y, returns a tuple of tensors
        # (x-values, y-values) each of shape (numSamples, numSamples)
        self.data_in = torch.cat((grid[0].reshape(-1,1),grid[1].reshape(-1,1)),1)
        # reshape data into shape (numSamples^2,2)

    def __len__(self):
        """
        Arguments:
        None
        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return self.data_in.shape[0]
    
    def __getitem__(self, i):
        """
        Used by DataLoader object to retrieve training data points

        Arguments:
        idx (int) -- index of data point required
        
        Returns:
        [x,y] (tensor shape (1,2)) -- data point at index 'idx'
        """
        return self.data_in[i]
    

class UniformDataSet(torch.utils.data.Dataset):
    """
    An object which generates uniformly sampled (x,y) values for the input node 
    """
    def __init__(self, xRange, yRange, numSamples):
        """
        Arguments:
        xRange (list of length 2) -- lower and upper limits for input values x
        yRange (list of length 2) -- lower and upper limits for input values y
        numSamples (int) -- number of training data samples along each axis

        Returns:
        DataSet object with one attributes:
            dataIn (PyTorch tensor of shape (numSamples^2,2)) -- 'numSamples'^2
                uniformly sampled grid points from (xRange[0], yRange[0]) to (xRange[1], yRange[1])
        """
        # uniformly sample numSamples^2 x-values in xRange
        X  = torch.distributions.Uniform(xRange[0],xRange[1]).sample((int(numSamples**2),1))
        X.requires_grad = True
        # uniformly sample numSamples^2 y-values in yRange
        Y  = torch.distributions.Uniform(yRange[0],yRange[1]).sample((int(numSamples**2),1))
        Y.requires_grad = True

        # format these 100 (x,y) coordinates in a tensor of shape (numSamples^2,2)
        self.data_in = torch.cat((X,Y),1)

    def __len__(self):
        """
        Arguments:
        None
        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return self.data_in.shape[0]
    
    def __getitem__(self, i):
        """
        Used by DataLoader object to retrieve training data points

        Arguments:
        idx (int) -- index of data point required
        
        Returns:
        [x,y] (tensor shape (1,2)) -- data point at index 'idx'
        """
        return self.data_in[i]

class PDESolver(torch.nn.Module):
    """
    The neural network object, with 2 nodes in the input layer,
    1 node in the output layer, and 1 hidden layer with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layer

        Returns:
        PDESolver object (neural network) with two attributes:
        fc1 (fully connected layer) -- linear transformation of hidden layer
        fc2 (fully connected layer) -- linear transformation of outer layer
        """
        super(PDESolver, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 2, out_features = numHiddenNodes)
        self.fc2 = torch.nn.Linear(in_features = numHiddenNodes, out_features = 1)

    def forward(self, input):
        """
        Function which connects inputs to outputs in the neural network.

        Arguments:
        input (PyTorch tensor shape (batchSize,2)) -- input of neural network

        Returns:
        z (PyTorch tensor shape (batchSize,1)) -- output of neural network
        """
        # tanh activation function used on hidden layer
        h = torch.tanh(self.fc1(input))
        # Linear activation function used on outer layer
        z = self.fc2(h)
        return z

def trial_term(x,y):
    """
    First term B(x,y) in trial solution that helps to satisfy BCs
    f(0,y) = 0, f(1,y) = 0, f(x,0) = 0, f_{y}(x,1) = 2*sin(pi*x)
    B(x,y) = y * 2 * sin(pi*x)
    """
    return 2 * y * torch.sin(np.pi * x)

def trial(x,y,n_outXY,n_outX1,n_outX1_y):
    return trial_term(x,y) + x*(1-x)*y*(n_outXY - n_outX1 - n_outX1_y)

def dx_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy):
    """
    f_x = 2*y*pi*cos(pi*x) + 
                y * [(1-2*x)(N - N(x,1) - N_{y}(x,1)) 
                    + x(1-x)(N_{x} - N_{x}(x,1) - N_{xy}(x,1)]
    """
    return ( 2* y *np.pi * torch.cos(np.pi*x) 
            + y * ((1-2*x) * (n_outXY - n_outX1 - n_outX1_y)
                + x*(1-x)*(n_outXY_x - n_outX1_x - n_outX1_xy)))

def dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy):
    """
    f_xx = -2*y*pi^2*sin(pi*x) 
                + y [ (-2)*(N - N(x,1) - N_{y}(x,1)
                        + 2(1-2x)((N_{x} - N_{x}(x,1) - N_{xy}(x,1))
                        + x(1-x)(N_{xx} - N_{xx}(x,1) - N_{xxy}(x,1))]"""
    return ( -2* y *(np.pi)**2 * torch.sin(np.pi*x) 
            + y * ( (-2) * (n_outXY - n_outX1 - n_outX1_y)
                + 2*(1-2*x) * (n_outXY_x - n_outX1_x - n_outX1_xy)
                + x*(1-x)*(n_outXY_xx - n_outX1_xx - n_outX1_xxy)))

def dy_trial(x,y, n_outXY, n_outX1, n_outX1_y, n_outXY_y):
    """
    f_y = 2sin(pi*x) + 
            + x(1-x)[(N - N(x,1) - N_{y}(x,1)) + y * N_{y}]
    """
    return (2*torch.sin(np.pi *x) + x*(1-x) *
        ((n_outXY - n_outX1 - n_outX1_y) + (y* n_outXY_y)))

def dy2_trial(x,y,n_outXY_y,n_outXY_yy):
    """
    f_yy = x(1-x)[2N_{y} + y * N_{yy}]
    """
    return (x * (1-x) * (2 * n_outXY_y + y * n_outXY_yy))

def diffEq(x,y,trial_dx2,trial_dy2):
    RHS = (2-((np.pi*y)**2)) * torch.sin(np.pi * x)
    return trial_dx2 + trial_dy2 - RHS

def solution(x, y):
    return (y**2) * torch.sin(np.pi * x)


def train(network, loader, lossFn, optimiser,numEpochs):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for _ in range(numEpochs):
        for batch in loader:
            x, y = torch.split(batch,1, dim=1)
            y_ones = torch.ones_like(y)
            # Coordinates (x,1) for all x in batch
            x1 = torch.cat((x,y_ones),1)

            # Neural network output at (x,y)
            n_outXY = network(batch)
            # Neural network output at (x,1)
            n_outX1 = network(x1)

            # Get all required derivatives of n(x,y)
            dn = grad(n_outXY, batch, torch.ones_like(n_outXY), retain_graph=True, create_graph=True)[0]
            # n_x , n_y
            n_outXY_x, n_outXY_y = dn[:,0].view(-1,1), dn[:,1].view(-1,1)
            dn2 = grad(dn, batch, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
            # n_xx , n_yy
            n_outXY_xx, n_outXY_yy = torch.split(dn2 , 1, dim = 1 )

            # Get all required derivatives of n(x,1):
            dn_x1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
            # n_x |(y=1) , n_y |(y=1)
            n_outX1_x, n_outX1_y = torch.split(dn_x1 , 1, dim = 1 )

            dn2dx_x1 = grad(n_outX1_x, x1, torch.ones_like(n_outX1_x), retain_graph = True, create_graph = True)[0]
            # n_xx |(y=1)
            n_outX1_xx , _= torch.split(dn2dx_x1, 1, dim = 1)

            dn2dy_x1 = grad(n_outX1_y, x1, torch.ones_like(n_outX1_y), retain_graph = True, create_graph = True)[0]
            # n_xy |(y=1)
            n_outX1_xy , _ = torch.split(dn2dy_x1, 1, dim=1)

            dn3dy_x1 = grad(n_outX1_xy, x1, torch.ones_like(n_outX1_xy), retain_graph = True, create_graph = True)[0]
            # n_xxy |(y=1)
            n_outX1_xxy ,  _= torch.split(dn3dy_x1, 1, dim=1)
            
            # Get second derivatives of trial solution
            trial_dx2 = dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy)
            trial_dy2 = dy2_trial(x,y,n_outXY_y,n_outXY_yy)
            
            # Calculate LHS of differential equation D(x,y) = 0
            D = diffEq(x,y,trial_dx2,trial_dy2)

            # calculate cost
            cost = lossFn(D, torch.zeros_like(D))
        
            # Optimization algorithm
            cost.backward() # perform backpropagation
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero

        cost_list.append(cost.item()) # store final cost of every epoch

    network.train(False)
    return cost_list


def plotNetwork(network, epoch, samplingMethod):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    batch = UniformDataSet(xRange,yRange,numSamples).data_in
    x, y = torch.split(batch,1, dim=1)
    y_ones = torch.ones_like(y)
    # Coordinates (x,1) for all x in batch
    x1 = torch.cat((x,y_ones),1)

    # Neural network output at (x,y)
    n_outXY = network(batch)
    # Neural network output at (x,1)
    n_outX1 = network(x1)

    # Get all required derivatives of n(x,y)
    dn = grad(n_outXY, batch, torch.ones_like(n_outXY), retain_graph=True, create_graph=True)[0]
    # n_x , n_y
    n_outXY_x, n_outXY_y = dn[:,0].view(-1,1), dn[:,1].view(-1,1)
    dn2 = grad(dn, batch, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
    # n_xx , n_yy
    n_outXY_xx, n_outXY_yy = torch.split(dn2 , 1, dim = 1 )

    # Get all required derivatives of n(x,1):
    dn_x1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
    # n_x |(y=1) , n_y |(y=1)
    n_outX1_x, n_outX1_y = torch.split(dn_x1 , 1, dim = 1 )

    dn2dx_x1 = grad(n_outX1_x, x1, torch.ones_like(n_outX1_x), retain_graph = True, create_graph = True)[0]
    # n_xx |(y=1)
    n_outX1_xx , _= torch.split(dn2dx_x1, 1, dim = 1)

    dn2dy_x1 = grad(n_outX1_y, x1, torch.ones_like(n_outX1_y), retain_graph = True, create_graph = True)[0]
    # n_xy |(y=1)
    n_outX1_xy , _ = torch.split(dn2dy_x1, 1, dim=1)

    dn3dy_x1 = grad(n_outX1_xy, x1, torch.ones_like(n_outX1_xy), retain_graph = True, create_graph = True)[0]
    # n_xxy |(y=1)
    n_outX1_xxy ,  _= torch.split(dn3dy_x1, 1, dim=1)
    
    # Get second derivatives of trial solution
    trial_dx2 = dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy)
    trial_dy2 = dy2_trial(x,y,n_outXY_y,n_outXY_yy)
    
    # Calculate LHS of differential equation D(x,y) = 0
    D = diffEq(x,y,trial_dx2,trial_dy2)

    # calculate cost
    cost = lossFn(D, torch.zeros_like(D))
    print("test cost = ", cost.item())

    # Get trial solution
    trialSolution = trial(x,y,n_outXY,n_outX1,n_outX1_y)

    # Get exact solution
    exact = solution(x,y).detach().numpy()

    # Calculate residual error
    trialSolution = trialSolution.detach().numpy()
    surfaceLoss = ((trialSolution-exact)**2).mean()
    print("trial-solution error = ", surfaceLoss)

    # PLOT SURFACE

    x_lin  = torch.linspace(xRange[0],xRange[1],numSamples, requires_grad=True)
    y_lin  = torch.linspace(yRange[0],yRange[1],numSamples, requires_grad=True)
    X,Y = torch.meshgrid(x_lin,y_lin)
    x, y  = X.reshape(-1,1), Y.reshape(-1,1)

    # Get network output at (x,y)
    xy = torch.cat((x,y),1)
    y_ones = torch.ones_like(y)
    # Coordinates (x,1) for all x in batch
    x1 = torch.cat((x,y_ones),1)

    # Neural network output at (x,y)
    n_outXY = network(xy)
    # Neural network output at (x,1)
    n_outX1 = network(x1)

    # Get all required derivatives of n(x,1):
    dn_x1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
    # n_x |(y=1) , n_y |(y=1)
    n_outX1_x, n_outX1_y = torch.split(dn_x1 , 1, dim = 1 )

    # Get trial solution
    trialSolution = trial(x,y,n_outXY,n_outX1,n_outX1_y)
    # Get exact solution
    exact = solution(x,y).detach().numpy()
    trialSolution = trialSolution.reshape(numSamples,numSamples).detach().numpy()

    # Plot trial and exact solutions
    X = X.detach().numpy()
    Y = Y.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,Y,trialSolution,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(X,Y,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(fontsize = 16)
    ax.set_title("Sampling Method: " + samplingMethod + ", " + str(epoch) + " Epochs" , fontsize = 16)
    plt.show()

    return surfaceLoss

numSamples  = 10
xRange      = [0,1]
yRange      = [0,1]
numEpochs   = 1000
totalEpochs = 5000
networkDict = {}

datasetDict = {"Uniform" : UniformDataSet(xRange,yRange,numSamples), 
               "Lattice" : LinearDataSet(xRange,yRange,numSamples)}

for samplingMethod in datasetDict:
    networkDict = {}
    trainData = datasetDict[samplingMethod]
    try: # load saved network if possible
        checkpoint = torch.load('problem7InitialNetwork.pth')
        network    = checkpoint['network']
    except: # create new network
        network    = PDESolver(numHiddenNodes=16)
        checkpoint = {'network': network}
        torch.save(checkpoint, 'problem7InitialNetwork.pth')

    lossFn      = torch.nn.MSELoss()
    optimiser   = torch.optim.Adam(network.parameters(), lr = 1e-3)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples**2), shuffle=True)

    epoch = 0 
    costList = []

    while epoch < totalEpochs:
        costList.extend(train(network, trainLoader, lossFn, optimiser, numEpochs))
        epoch += numEpochs
    
    print(f"{epoch} epochs total, final cost = {costList[-1]}")

    plt.semilogy(costList)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Cost", fontsize = 16)
    plt.title(f"Training Costs, Sampling Method = {samplingMethod}", fontsize = 16)
    plt.show()

    plotNetwork(network, epoch, samplingMethod)
    
    networkDict["costList"] = costList
    networkDict["network"] = network
    networkDict["dataSet"] = trainData
    torch.save(networkDict, 'problem7' + samplingMethod + '.pth')
#%%