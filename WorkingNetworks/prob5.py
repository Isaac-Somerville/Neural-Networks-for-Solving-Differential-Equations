#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad


class DataSet(torch.utils.data.Dataset):
    """
    An object which generates the (x,y) values for the input node 
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

def solution(x,y):
    """solution to Lagaris problem 5"""
    return torch.exp(-x) * (x + y**3)

def trial_term(x,y):
    """
    First term in trial solution that helps to satisfy BCs
    """
    e_inv = np.exp(-1)
    return ((1-x)*(y**3) + x*(1+(y**3))*e_inv + (1-y)*x*(torch.exp(-x)-e_inv) +
            y * ((1+x)*torch.exp(-x) - (1-x+(2*x*e_inv))))

def trial(x,y,n_out):
    """Trial solution f(x,y) to Lagaris problem 5"""
    return trial_term(x,y) + x*(1-x)*y*(1-y)*n_out

def dx_trial(x,y,n_out,n_x):
    """f_x(x,y)"""
    return (-torch.exp(-x) * (x+y-1) - y**3 + (y**2 +3) * y * np.exp(-1) + y
            + y*(1-y)* ((1-2*x) * n_out + x*(1-x)*n_x))

def dx2_trial(x,y,n_out,n_x,n_xx):
    """f_xx(x,y)"""
    return (torch.exp(-x) * (x+y-2)
            + y*(1-y)* ((-2*n_out) + 2*(1-2*x)*n_x) + x*(1-x)*n_xx)

def dy_trial(x,y,n_out, n_y):
    """f_y(x,y)"""
    return (3*x*(y**2 +1) *np.exp(-1) - (x-1)*(3*(y**2)-1) + torch.exp(-x)
            + x*(1-x)* ((1-2*y) * n_out + y*(1-y)*n_y) )

def dy2_trial(x,y,n_out,n_y,n_yy):
    """f_yy(x,y)"""
    return (np.exp(-1) * 6 * y * (-np.exp(1)*x + x + np.exp(1))
            + x*(1-x)* ((-2*n_out) + 2*(1-2*y)*n_y) + y*(1-y)*n_yy)

def diffEq(x,y,trial_dx2,trial_dy2):
    """Differential equation from Lagaris problem 5"""
    RHS = torch.exp(-x) * (x - 2 + y**3 + 6*y)
    return trial_dx2 + trial_dy2 - RHS


def train(network, loader, lossFn, optimiser, numEpochs):
    """
    A function to train a neural network to solve a 
    2-dimensional PDE with Dirichlet boundary conditions

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
    for epoch in range(numEpochs):
        for batch in loader:
            n_out = network(batch) # network output

            dn = grad(n_out, batch, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            dn2 = grad(dn, batch, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
            
            # Get first partial derivatives of neural network output: n_x , n_y
            n_x, n_y = torch.split(dn, split_size_or_sections=1, dim=1)
            # Get second derivatives of neural network output: n_xx,  n_yy
            n_xx, n_yy = torch.split(dn2, split_size_or_sections=1, dim=1)

            x, y = torch.split(batch, 1, dim=1) # separate batch into x- and y-values
            # Get second derivatives of trial solution: f_{xx}(x,y) and f_{yy}(x,y)
            trial_dx2 = dx2_trial(x,y,n_out,n_x,n_xx)
            trial_dy2 = dy2_trial(x,y,n_out,n_y,n_yy)
            # Get value of LHS of differential equation D(x,y) = 0
            D = diffEq(x,y, trial_dx2, trial_dy2)

            # Calculate and store cost
            cost = lossFn(D, torch.zeros_like(D))
        
            # Optimization algorithm
            cost.backward() # perform backpropagation
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero
            
        cost_list.append(cost.item())
    network.train(False)
    return cost_list

def plotNetwork(network, epoch):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    numTestSamples = 12
    X  = torch.linspace(xRange[0],xRange[1],numTestSamples, requires_grad=True)
    Y  = torch.linspace(yRange[0],yRange[1],numTestSamples, requires_grad=True)
    x_mesh,y_mesh = torch.meshgrid(X,Y)

    # Format input into correct shape
    input = torch.cat((x_mesh.reshape(-1,1),y_mesh.reshape(-1,1)),1)

    # Get output of neural network
    N = network.forward(input)

    dn = grad(N, input, torch.ones_like(N), retain_graph=True, create_graph=True)[0]
    dn2 = grad(dn, input, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
    
    # Get first derivatives of NN output: dn/dx , dn/dy
    n_x, n_y = torch.split(dn, split_size_or_sections=1, dim=1)

    # Get second derivatives of NN output: d^2 n / dx^2,  d^2 n / dy^2
    n_xx, n_yy = torch.split(dn2, split_size_or_sections=1, dim=1)

    x, y = torch.split(input, split_size_or_sections=1, dim=1)
    # Get value of trial solution f_{xx}(x,y) and f_{yy}(x,y)
    trial_dx2 = dx2_trial(x,y,N,n_x,n_xx)
    trial_dy2 = dy2_trial(x,y,N,n_y,n_yy)

    # Get value of diff equations D(x) = 0
    D = diffEq(x,y, trial_dx2, trial_dy2)

    # Calculate and store cost
    cost = lossFn(D, torch.zeros_like(D))
    print("test cost = ", cost.item())

    # Get trial solution, put into correct shape
    output = trial(x_mesh.reshape(-1,1),y_mesh.reshape(-1,1),N)
    output = output.reshape(numTestSamples,numTestSamples).detach().numpy()

    # Get exact solution
    exact = solution(x_mesh,y_mesh).detach().numpy()

    # Calculate residual error
    surfaceError = ((output-exact)**2).mean()
    print("mean square difference between trial and exact solution = ", surfaceError ) 

    # Plot both trial and exact solutions
    x_mesh = x_mesh.detach().numpy()
    y_mesh = y_mesh.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x_mesh,y_mesh,output,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(x_mesh,y_mesh,exact, label = 'Exact Solution')
    ax.set_xlabel('x', fontsize = 16)
    ax.set_ylabel('y', fontsize = 16)
    ax.set_zlabel('z', fontsize = 16)
    ax.legend()
    ax.set_title("Learning Rate = " + str(lr) + ": " + str(epoch) + " Epochs", fontsize = 16)
    #ax.view_init(30, 315)
    plt.show()
    return surfaceError, cost.item()

testCosts = []
surfaceErrors = []

# learningRates = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
learningRates = [1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3, 7e-3, 8e-3, 9e-3, 1e-2]

try: # load saved network if possible
    checkpoint = torch.load('problem5InitialNetwork.pth')
    network    = checkpoint['network']
except: # create new network
    network    = PDESolver(numHiddenNodes=16)
    checkpoint = {'network': network}
    torch.save(checkpoint, 'problem5InitialNetwork.pth')

xRange = [0,1]
yRange = [0,1]
numSamples = 10
numEpochs = 1000
costListDict = {}

lossFn      = torch.nn.MSELoss()
trainSet    = DataSet(xRange,yRange,numSamples)
trainLoader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=int(numSamples**2), shuffle=True)

for lr in learningRates:
    checkpoint = torch.load('problem5InitialNetwork.pth')
    network    = checkpoint['network'] # reset network to initial values
    optimiser  = torch.optim.Adam(network.parameters(), lr = lr)
    costList = []
    epoch = 0
    totalEpochs = 10000
    while epoch < totalEpochs:
        costList.extend(train(network, trainLoader, lossFn, optimiser, numEpochs))
        epoch += numEpochs
    
    print("lr = ", lr)
    print(f"{epoch} epochs total, final cost = {costList[-1]}")

    plt.semilogy(costList)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Cost", fontsize = 16)
    plt.title(f"Training Cost, Learning Rate = {lr}", fontsize = 16)
    plt.show()

    costListDict[lr] = costList
    surfaceError, testCost = plotNetwork(network, epoch)
    surfaceErrors.append(surfaceError)
    testCosts.append(testCost)

# for lr in learningRates:
#     plt.semilogy(costListDict[lr], label = "Learning Rate: " + str(lr))
#     if lr == 1e-3 or lr == 1e-1:
#         plt.xlabel("Epochs", fontsize = 16)
#         plt.ylabel("Training Cost", fontsize = 16)
#         plt.legend(loc = "upper right", fontsize = 16)
#         plt.title("Effect of Learning Rate on Training Costs", fontsize = 16)
#         plt.show()


torch.save(costListDict, "prob5CostLists.pth")
plt.semilogy(learningRates,surfaceErrors)
plt.xlabel("Learning Rate", fontsize = 16)
plt.ylabel("Final Trial-Solution Error ", fontsize = 16)
plt.title("Effect of Learning Rate on Trial-Solution Error", fontsize = 16)
plt.show()

plt.semilogy(learningRates,testCosts)
plt.xlabel("Learning Rate", fontsize = 16)
plt.ylabel("Final Cost", fontsize = 16)
plt.title("Effect of Learning Rate on Final Cost", fontsize = 16)
plt.show()

#%%