#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import time

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

        # format these numSamples^2 (x,y) coordinates in a tensor of shape (numSamples^2,2)
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
    

class NormalDataSet(torch.utils.data.Dataset):
    """
    An object which generates Normally sampled (x,y) values for the input node 
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
                Normally-sampled grid points from (xRange[0], yRange[0]) to (xRange[1], yRange[1])
        """
        # sample numSamples^2 x-values in xRange from N(mu, sigma), where mu is the midpoint of xRange
        # and sigma is 1/9  of the width of xRange: this means all values are virtually guaranteed to lie in xRange
        # for different value of numSamples, adjust denominator of sigma
        X  = torch.normal((xRange[1]- xRange[0])/2, (xRange[1]- xRange[0])/9, (int(numSamples**2),1))
        X.requires_grad = True
        # same process for y-values
        Y  = torch.normal((yRange[1]- yRange[0])/2, (yRange[1]- yRange[0])/9, (int(numSamples**2),1))
        Y.requires_grad = True
        # format these numSamples^2 (x,y) coordinates in a tensor of shape (numSamples^2,2)
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
    First term F(x,y) in trial solution that helps to satisfy BCs f(0,y) = f(1,y) = f(x,0) = 0, f_{y}(x,1) = 2*sin(pi*x)
    F(x,y) = y * 2 * sin(pi*x)

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    Returns:
    y * 2 * sin(pi*x) (PyTorch tensor shape (batchSize,1)) -- value of first term F(x,y) in trial solution
    """
    return y**2 * torch.sin(np.pi * x)

def trial(x,y,n_outXY,n_outX1,n_outX1_y):
    """
    Trial solution to Lagaris problem 8: y * 2 * sin(pi*x) + x*(1-x)*y*[N(x,y) - N(x,1) - N_{y}(x,1)]

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    n_outXY (PyTorch tensor shape (batchSize,1)) -- N(x,y), neural network outputs at (x,y)
    n_outX1 (PyTorch tensor shape (batchSize,1)) -- N(x,1), neural network outputs at (x,1)
    n_outX1_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,1) partial derivative w.r.t. y of neural network at (x,1)
    Returns:
    f(x,y) (PyTorch tensor shape (batchSize,1)) -- trial solution at (x,y)
    """
    return trial_term(x,y) + x*(1-x)*y*(n_outXY - n_outX1 - n_outX1_y)

def dx_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy):
    """
    First derivative w.r.t. x of trial solution at (x,y):
    f_{x}(x,y) = y**2*pi*cos(pi*x) + y * [(1-2*x)(N - N(x,1) - N_{y}(x,1)) + x(1-x)(N_{x} - N_{x}(x,1) - N_{xy}(x,1)]

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    n_outXY (PyTorch tensor shape (batchSize,1)) -- N(x,y), neural network outputs at (x,y)
    n_outX1 (PyTorch tensor shape (batchSize,1)) -- N(x,1), neural network outputs at (x,1)
    n_outX1_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,1)
    n_outXY_x (PyTorch tensor shape (batchSize,1)) -- N_{x}(x,y)
    n_outX1_x (PyTorch tensor shape (batchSize,1)) --  N_{x}(x,1)
    n_outX1_xy (PyTorch tensor shape (batchSize,1)) --  N_{xy}(x,1)
    Returns:
    f_{x}(x,y) (PyTorch tensor shape (batchSize,1)) -- first derivative w.r.t. x of trial solution at (x,y)"""
    return ( y**2 *np.pi * torch.cos(np.pi*x) + y * ((1-2*x) * (n_outXY - n_outX1 - n_outX1_y) 
                + x*(1-x)*(n_outXY_x - n_outX1_x - n_outX1_xy)))

def dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy):
    """
    Second derivative w.r.t. x of trial solution at (x,y):
    f_{xx}(x,y) = -y**2*pi^2*sin(pi*x) + y [ (-2)*(N - N(x,1) - N_{y}(x,1) + 2(1-2x)((N_{x} - N_{x}(x,1) - N_{xy}(x,1))
                    + x(1-x)(N_{xx} - N_{xx}(x,1) - N_{xxy}(x,1))]

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    n_outXY (PyTorch tensor shape (batchSize,1)) -- N(x,y), neural network outputs at (x,y)
    n_outX1 (PyTorch tensor shape (batchSize,1)) -- N(x,1), neural network outputs at (x,1)
    n_outX1_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,1)
    n_outXY_x (PyTorch tensor shape (batchSize,1)) -- N_{x}(x,y)
    n_outX1_x (PyTorch tensor shape (batchSize,1)) --  N_{x}(x,1)
    n_outX1_xy (PyTorch tensor shape (batchSize,1)) -- N_{xy}(x,1)
    n_outXY_xx (PyTorch tensor shape (batchSize,1)) -- N_{xx}(x,y)
    n_outX1_xx (PyTorch tensor shape (batchSize,1)) -- N_{xx}(x,1)
    n_outX1_xxy (PyTorch tensor shape (batchSize,1)) -- N_{xxy}(x,1)   

    Returns:
    f_{xx}(x,y) (PyTorch tensor shape (batchSize,1)) -- second derivative w.r.t. x of trial solution at (x,y)      
    """
    return ( -y**2 *(np.pi)**2 * torch.sin(np.pi*x) + y * ( (-2) * (n_outXY - n_outX1 - n_outX1_y)
                + 2*(1-2*x) * (n_outXY_x - n_outX1_x - n_outX1_xy) + x*(1-x)*(n_outXY_xx - n_outX1_xx - n_outX1_xxy)))

def dy_trial(x,y, n_outXY, n_outX1, n_outX1_y, n_outXY_y):
    """
    First derivative w.r.t. y of trial solution at (x,y):
    f_{y}(x,y) = 2ysin(pi*x) + x(1-x)[(N(x,y) - N(x,1) - N_{y}(x,1)) + y * N_{y}(x,y)]

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    n_outXY (PyTorch tensor shape (batchSize,1)) -- N(x,y), neural network outputs at (x,y)
    n_outX1 (PyTorch tensor shape (batchSize,1)) -- N(x,1), neural network outputs at (x,1)
    n_outX1_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,1)
    n_outXY_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,y)

    Returns:
    f_{y}(x,y) (PyTorch tensor shape (batchSize,1)) -- first derivative w.r.t. y of trial solution at (x,y)
    """
    return (2*y*torch.sin(np.pi *x) + x*(1-x) * ((n_outXY - n_outX1 - n_outX1_y) + (y* n_outXY_y)))

def dy2_trial(x,y,n_outXY_y,n_outXY_yy):
    """
    Second derivative w.r.t. y of trial solution at (x,y): 
    f_{yy}(x,y) = 2sin(pi*x) + x(1-x)[2N_{y}(x,y) + y * N_{yy}(x,y)]
    
    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    n_outXY_y (PyTorch tensor shape (batchSize,1)) -- N_{y}(x,y)
    n_outXY_yy (PyTorch tensor shape (batchSize,1)) -- N_{yy}(x,y)

    Returns:
    f_{yy}(x,y) (PyTorch tensor shape (batchSize,1)) -- second derivative w.r.t. y of trial solution at (x,y)
    """
    return (2*torch.sin(np.pi *x) + x * (1-x) * (2 * n_outXY_y + y * n_outXY_yy))

def diffEq(x,y,trialFunc, trial_dy, trial_dx2, trial_dy2):
    """
    Returns D(x,y) from differential equation D(x,y) = 0, Lagaris problem 8

    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs
    trialFunc (PyTorch tensor shape (batchSize,1)) -- f(x,y) trial solution at (x,y)
    trial_dy (PyTorch tensor shape (batchSize,1)) -- f_{y}(x,y)
    trial_dx2 (PyTorch tensor shape (batchSize,1)) -- f_{x}(x,y)
    trial_dy2 (PyTorch tensor shape (batchSize,1)) -- f_{yy}(x,y)

    Returns:
    D(x,y) (PyTorch tensor shape (batchSize,1)) -- D(x,y) from DE D(x,y) = 0, Lagaris problem 8
    """
    RHS = torch.sin(np.pi*x)*(2 - np.pi**2*y**2 + 2*y**3*torch.sin(np.pi*x))
    return trial_dx2 + trial_dy2 + trialFunc * trial_dy - RHS

def solution(x, y):
    """
    Analytic solution to Lagaris problem 8, f(x,y) = (y**2) * torch.sin(np.pi * x)
    
    Arguments:
    x (PyTorch tensor shape (batchSize,1)) -- x-values of neural network inputs
    y (PyTorch tensor shape (batchSize,1)) -- y-values of neural network inputs

    Returns:
    (y**2) * torch.sin(np.pi * x) (PyTorch tensor shape (batchSize,1)) -- true solution at (x,y)
    """
    return (y**2) * torch.sin(np.pi * x)


def train(network, loader, lossFn, optimiser, numEpochs):
    """
    A function to train a neural network to solve a 2-dimensional PDE with mixed boundary conditions

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
    network.train(True)
    for _ in range(numEpochs):
        if samplingMethod == "Uniform": # sample new points uniformly every epoch
            trainData = UniformDataSet(xRange,yRange,numSamples)
            loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples**2), shuffle=True)
        if samplingMethod == "Normal": # sample new points from Normal distribution every epoch
            trainData = NormalDataSet(xRange,yRange,numSamples)
            loader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples**2), shuffle=True)
        for batch in loader:
            x, y = torch.split(batch,1, dim=1) # separate batch into x- and y-values
            y_ones = torch.ones_like(y)     # create tensor of ones 
            x1 = torch.cat((x,y_ones),1)    # Coordinates (x,1) for all x in batch

            n_outXY = network(batch)    # Neural network output at (x,y)
            n_outX1 = network(x1)       # Neural network output at (x,1)

            # Get all required derivatives of n(x,y)
            grad_n_outXY = grad(n_outXY, batch, torch.ones_like(n_outXY), retain_graph=True, create_graph=True)[0]
            n_outXY_x, n_outXY_y = torch.split(grad_n_outXY,1,dim=1) # n_x , n_y

            grad_grad_n_outXY = grad(grad_n_outXY, batch, torch.ones_like(grad_n_outXY), retain_graph=True, create_graph=True)[0]
            n_outXY_xx, n_outXY_yy = torch.split(grad_grad_n_outXY , 1, dim = 1 ) # n_xx , n_yy

            # Get all required derivatives of n(x,1):
            grad_n_outX1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
            n_outX1_x, n_outX1_y = torch.split(grad_n_outX1 , 1, dim = 1 )     # n_x |(y=1) , n_y |(y=1)

            grad_n_outX1_x = grad(n_outX1_x, x1, torch.ones_like(n_outX1_x), retain_graph = True, create_graph = True)[0]
            n_outX1_xx , n_outX1_xy = torch.split(grad_n_outX1_x, 1, dim = 1)     # n_xx |(y=1), n_xy |(y=1)

            grad_n_outX1_xy = grad(n_outX1_xy, x1, torch.ones_like(n_outX1_xy), retain_graph = True, create_graph = True)[0]
            n_outX1_xxy ,  _= torch.split(grad_n_outX1_xy, 1, dim=1)     # n_xxy |(y=1)
            
            # Get trial solution
            trialFunc = trial(x, y, n_outXY, n_outX1, n_outX1_y)
            # Get first partial derivative (w.r.t y) of trial solution
            trial_dy  = dy_trial(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_y)
            # Get second partial derivatives of trial solution
            trial_dx2 = dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, 
                                  n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy)
            trial_dy2 = dy2_trial(x,y,n_outXY_y,n_outXY_yy)
            
            # Calculate LHS of differential equation D(x,y) = 0
            D = diffEq(x, y, trialFunc, trial_dy, trial_dx2, trial_dy2)

            cost = lossFn(D, torch.zeros_like(D))   # calculate cost
            cost.backward()     # perform backpropagation
            optimiser.step()    # perform parameter optimisation
            optimiser.zero_grad()   # reset gradients to zero

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
    grad_n_outXY = grad(n_outXY, batch, torch.ones_like(n_outXY), retain_graph=True, create_graph=True)[0]
    # n_x , n_y
    n_outXY_x, n_outXY_y = grad_n_outXY[:,0].view(-1,1), grad_n_outXY[:,1].view(-1,1)
    grad_grad_n_outXY = grad(grad_n_outXY, batch, torch.ones_like(grad_n_outXY), retain_graph=True, create_graph=True)[0]
    # n_xx , n_yy
    n_outXY_xx, n_outXY_yy = torch.split(grad_grad_n_outXY , 1, dim = 1 )

    # Get all required derivatives of n(x,1):
    grad_n_outX1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
    # n_x |(y=1) , n_y |(y=1)
    n_outX1_x, n_outX1_y = torch.split(grad_n_outX1 , 1, dim = 1 )

    grad_n_outX1_x = grad(n_outX1_x, x1, torch.ones_like(n_outX1_x), retain_graph = True, create_graph = True)[0]
    # n_xx |(y=1),  n_xy |(y=1)
    n_outX1_xx , n_outX1_xy = torch.split(grad_n_outX1_x, 1, dim = 1)

    # grad_n_outX1_y = grad(n_outX1_y, x1, torch.ones_like(n_outX1_y), retain_graph = True, create_graph = True)[0]
    # # 
    # n_outX1_xy , _ = torch.split(grad_n_outX1_y, 1, dim=1)

    grad_n_outX1_xy = grad(n_outX1_xy, x1, torch.ones_like(n_outX1_xy), retain_graph = True, create_graph = True)[0]
    # n_xxy |(y=1)
    n_outX1_xxy ,  _= torch.split(grad_n_outX1_xy, 1, dim=1)
    
    # Get trial solution
    trialFunc = trial(x, y, n_outXY, n_outX1, n_outX1_y)
    # Get first derivative of trial solution
    trial_dy  = dy_trial(x, y, n_outXY, n_outX1, n_outX1_y, n_outXY_y)
    # Get second derivatives of trial solution
    trial_dx2 = dx2_trial(x,y,n_outXY, n_outX1, n_outX1_y, n_outXY_x, n_outX1_x, n_outX1_xy, n_outXY_xx, n_outX1_xx, n_outX1_xxy)
    trial_dy2 = dy2_trial(x,y,n_outXY_y,n_outXY_yy)
    
    # Calculate LHS of differential equation D(x,y) = 0
    D = diffEq(x, y, trialFunc, trial_dy, trial_dx2, trial_dy2)

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
    grad_n_outX1 = grad(n_outX1, x1, torch.ones_like(n_outX1), retain_graph=True, create_graph=True)[0]
    # n_x |(y=1) , n_y |(y=1)
    n_outX1_x, n_outX1_y = torch.split(grad_n_outX1 , 1, dim = 1 )

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
    ax.legend(fontsize = 12)
    ax.set_title(str(epoch) + " Epochs, " + "Sampling Method: " + samplingMethod, fontsize = 16)
    plt.show()

    return surfaceLoss

numSamples  = 10
xRange      = yRange    = [0,1]
numEpochs   = 1000
totalEpochs = 10000
networkDict = costListDict = {}

datasetDict = {"Normal" : NormalDataSet(xRange,yRange,numSamples),
               "Uniform" : UniformDataSet(xRange,yRange,numSamples), 
               "Lattice" : LinearDataSet(xRange,yRange,numSamples)}

for samplingMethod in datasetDict:
    networkDict = {}
    trainData = datasetDict[samplingMethod]
    
    x, y = torch.split(trainData.data_in, 1, 1)
    plt.plot(x.detach().numpy(),y.detach().numpy(), 'b.')
    plt.xlabel("x",fontsize = 16)
    plt.ylabel("y", fontsize = 16)
    plt.title("Data Points, Sampling Method: " + samplingMethod, fontsize = 16)
    plt.show()

    try: # load saved network if possible
        checkpoint = torch.load('problem8InitialNetwork.pth')
        network    = checkpoint['network']
    except: # create new network
        network    = PDESolver(numHiddenNodes=16)
        checkpoint = {'network': network}
        torch.save(checkpoint, 'problem8InitialNetwork.pth')

    lossFn      = torch.nn.MSELoss()
    optimiser   = torch.optim.Adam(network.parameters(), lr = 1e-3)
    trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples**2), shuffle=True)
    epoch = 0 
    costList = []

    start = time.time()
    while epoch < totalEpochs:
        costList.extend(train(network, trainLoader, lossFn, optimiser, numEpochs))
        epoch += numEpochs
    end = time.time()
    print("total training time = ", end-start, " seconds")

    costListDict[samplingMethod] = costList
    networkDict["costList"] = costList
    networkDict["network"] = network
    torch.save(networkDict, 'problem8' + samplingMethod + '.pth')
    
    print(f"{epoch} epochs total, final cost = {costList[-1]}")

    plt.semilogy(costList)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Cost", fontsize = 16)
    plt.title(f"Training Costs, Sampling Method: {samplingMethod}", fontsize = 16)
    plt.show()

    plotNetwork(network, epoch, samplingMethod)

for samplingMethod in costListDict:
    plt.semilogy(costListDict[samplingMethod], label = samplingMethod)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Training Cost", fontsize = 16)
plt.legend(loc = "upper right", fontsize = 16)
plt.title("Effect of Sampling Method on Training Costs", fontsize = 16)
plt.show()

#%%