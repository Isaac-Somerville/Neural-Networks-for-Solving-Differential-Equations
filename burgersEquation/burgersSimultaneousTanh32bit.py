#%%
#############
# Tanh activation function, train for u(x,t) and lambda_i simultaneously
# use 32-bit floats
#############

import torch
import torch.utils.data
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad
import scipy.io


class DataSet(torch.utils.data.Dataset):
    """Samples 'numSamples' random samples of (x, t, u(x,t)) training data from data set of 25,600"""
    def __init__(self, XT, u_exact, numSamples):
        """
        Arguments:
        XT (array of shape (25600, 2)) -- (x,t) input samples
        u_exact (array of shape (25600, 2)) -- exact values u(x,t) for training
        numSamples (int) -- number of training data samples required

        Returns:
        DataSet object with one attribute:
            dataIn (tuple of PyTorch tensors of shape (numSamples,2) and (numSamples,1)) -- 'numSamples'
                randomly sampled (x,t) points with their corresponding function values u(x,t)
        """
        # generate numSamples random indices to get training samples
        idx = np.random.choice(XT.shape[0], numSamples, replace=False) 

        # store (x,t) values and u(x,t) values in two separate tensors, and convert them to 32-bit
        XT_train = torch.tensor(XT[idx,:], requires_grad=True).float()
        u_train = torch.tensor(u_exact[idx,:], requires_grad=True).float()

        # input of forward function must have shape (batch_size, 2)
        # u-values for training must have shape (batch_size, 1)
        # we load this data as a tuple of tensors
        self.data_in = (XT_train, u_train)
        
    def __len__(self):
        """
        Arguments:
        None
        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return self.data_in[0].shape[0]
    
    def __getitem__(self, idx):
        """
        Used by DataLoader object to retrieve training data points

        Arguments:
        idx (int) -- index of data point required
        
        Returns:
        ((x,t) , u(x,t)) (tuple of tensors shape (1,2) and (1,1)) -- data point (x,t) and u(x,t) at index 'idx'
        """
        return (self.data_in[0][idx,:], self.data_in[1][idx])

class BurgersEquationSolver(torch.nn.Module):
    """
    A deep neural network object, with 2 nodes in the input layer, 1 node in the 
    output layer, and 'numHiddenLayers' hidden layers each with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes, numHiddenLayers):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layers
        numHiddenLayers (int) -- number of hidden layers
        Returns:
        BurgersEquationSolver object (neural network) with three attributes:
        fc1 (fully connected layer) -- linear transformation of first layer
        fcs (list of fully connected layers) -- linear transformations of hidden layers
        fcLast (fully connected layer) -- linear transformation of outer layer
        """
        super(BurgersEquationSolver, self).__init__()
        # create first layer with 2 inputs (x and t), apply Xavier initialisation
        self.fc1 = torch.nn.Linear(2, numHiddenNodes)
        self.fc1.apply(self.initWeightsXavier)
        # create list of hidden layers, apply Xavier initialisation
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers)])
        self.fcs.apply(self.initWeightsXavier)
        # create final layer with one output (u(x,t)), apply Xavier initialisation
        self.fcLast = torch.nn.Linear(numHiddenNodes, 1)
        self.fcLast.apply(self.initWeightsXavier)

        self.lmbda = torch.nn.Parameter(torch.rand(1)) # initialise parameter lambda
        self.nu = torch.nn.Parameter(torch.rand(1)) # initialise parameter nu

    def forward(self, input):
        """
        Function which performs forward propagation in the neural network.
        Arguments:
        input (PyTorch tensor shape (batchSize, 2)) -- input of neural network
        Returns:
        output (PyTorch tensor shape (batchSize, 1)) -- output of neural network
        """
        hidden = torch.tanh(self.fc1(input))
        # pass through all hidden layers
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        output = self.fcLast(hidden)
        return output
    
    def initWeightsXavier(self, layer):
        """
        Function which initialises weights according to Xavier initialisation
        Arguments:
        layer (Linear object) -- weights and biases of a layer
        Returns:
        None
        """
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight, gain = torch.nn.init.calculate_gain('tanh'))

def train(network, lossFn, optimiser, scheduler, loader, numEpochs):
    """
    A function to train a neural network to approximate the solution u(x,t) to Burger's equation,
    while simultaneously estimating the parameters lambda and nu from the equation

    Arguments:
    network (Module) -- the neural network
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    scheduler (Learning Rate Scheduler) -- reduces learning rate if cost value is plateauing
    loader (DataLoader) -- generates batches from the training dataset
    numEpochs (int) -- number of training epochs

    Returns:
    costList (list of length 'numEpochs') -- cost values of all epochs
    lambdaList (list of length 'numEpochs') -- lambda values of all epochs
    nuList (list of length 'numEpochs') -- nu values of all epochs
    """
    costList = []
    lmbdaList = []
    nuList = []
    network.train(True) # set network into training mode
    for _ in range(numEpochs):
        for batch in loader:
            input, batch_u_exact = batch # separate inputs (x,t) and exact values u(x,t)
            u_out = network.forward(input) # pass inputs (x,t) through network
            # calculate first- and second-order partial derivatives of network output
            du = grad(u_out, input, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
            d2u = grad(du, input, torch.ones_like(du), retain_graph=True, create_graph=True)[0]

            # separate partial derivatives to attain u_t, u_x, u_xx
            # use torch.split to preserve grad history
            u_x, u_t = torch.split(du, 1, dim =1)
            u_xx, u_tt = torch.split(d2u, 1, dim =1)

            # evaluate differential equation
            # since we know nu will always be positive, we train with exp(nu)
            diffEqLHS = u_t + (network.lmbda * u_out * u_x) - (torch.exp(network.nu) * u_xx)

            # calculate J_u and J_D, cost functions for approximation of u(x,t) and DE respectively
            uCost = lossFn(u_out, batch_u_exact)
            DECost = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))
            cost = uCost + DECost

            cost.backward() # perform back propagation
            optimiser.step() # update parameters
            optimiser.zero_grad() # reset gradients to zero
 
        scheduler.step(cost) # update scheduler, reduces learning rate if on plateau  
        # store cost, lambda- and nu-value of each epoch
        costList.append(cost.detach().numpy())
        lmbdaList.append(network.lmbda.item())
        nuList.append(torch.exp(network.nu).item())

    network.train(False) # set network out of training mode
    return costList, lmbdaList, nuList

def test(network, XT, u_exact, lossFn):
    """
    Tests network solution on all 25600 sample points
    """
    testData = DataSet(XT , u_exact, XT.shape[0])
    batch, batch_u_exact = testData.data_in
    u_out = network.forward(batch)

    du = grad(u_out, batch, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
    # print(du)

    d2u = grad(du, batch, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    # print(d2u)

    u_x, u_t = torch.split(du, 1, dim =1)
    # print(u_t)
    # print(u_x)

    u_xx, u_tt = torch.split(d2u, 1, dim =1)
    # print(u_xx)

    diffEqLHS = u_t + (network.lmbda * u_out * u_x) - (torch.exp(network.nu) * u_xx)

    # calculate costs for u, DE, lmbda and nu
    uTestLoss = lossFn(u_out, batch_u_exact)
    DETestLoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))
    lmbdaLoss = abs(network.lmbda - 1.) * 100
    nuCost = (abs(torch.exp(network.nu) - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
    print("u_test error = ", uTestLoss.item())
    print("DE_test error = ", DETestLoss.item())
    print("lmbda error = ", lmbdaLoss.item(), " %")
    print("nu error = ", nuCost.item(), " %")
    return


def plotNetwork(network, X, T, XT, u_exact, epoch):
    """
    Plots network solution for all 25600 sample points
    """
    XT = torch.tensor(XT).float()
    u_exact = torch.tensor(u_exact).float()

    # print(X)
    # print(T)
    u_out = network.forward(XT)
    # print(u_out)
    u_out = u_out.reshape(X.shape[0],X.shape[1])
    # print(u_out)
    u_out = u_out.detach().numpy()
    lmbda = network.lmbda
    nu = network.nu
    print("lmbda = ", lmbda.item())
    print("nu = ", torch.exp(nu).item())

    # print(X.shape)
    # print(T.shape)
    # print(u_out.shape)

    # Plot trial solution
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,T,u_out,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.set_xlabel('x', fontsize = 16)
    ax.set_ylabel('t', fontsize = 16)
    
    # Plot exact solution
    # ax.scatter(X,T,u_exact, label = 'Exact Solution')
    # ax.legend()

    ax.set_title("Burger's Equation: " + str(epoch) + " Epochs", fontsize = 16)
    plt.show()
    return

# load and format sample data (dictionary) for u(x,t), there are 25600 samples in total 
data = scipy.io.loadmat('burgersData.mat')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T
X, T = np.meshgrid(x,t)

XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_exact = Exact.flatten()[:,None]
numSamples = 2000 # number of training samples

try: # load saved network if possible
    checkpoint = torch.load('burgersSimultaneousTanh32bit.pth')
    epoch = checkpoint['epoch']
    network = checkpoint['network']
    trainData = checkpoint['trainData']
    optimiser = checkpoint['optimiser']
    scheduler = checkpoint['scheduler']
    costs = checkpoint['costs']
    lmbdas = checkpoint['lmbdas']
    nus = checkpoint['nus']
    print("model loaded")
except:
    try: # load initial state of model
        checkpoint  = torch.load('burgersTanhInitialNetwork.pth')
        network     = checkpoint['network']
        trainData   = checkpoint['trainData']
        print("initial model loaded")
    except:  # create new model and save its initial state
        network     = BurgersEquationSolver(numHiddenNodes=32, numHiddenLayers=8)
        trainData   = DataSet(XT, u_exact, numSamples)
        checkpoint  = {'network' : network, 'trainData' : trainData}
        torch.save(checkpoint, 'burgersTanhInitialNetwork.pth')
        print("new model created")
    epoch = 0
    optimiser = torch.optim.Adam(network.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor = 0.5, patience = 500, 
                    threshold = 1e-4, min_lr = 1e-6, verbose = True)
    costs, lmbdas, nus = [], [], []

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=numSamples, shuffle=True)
lossFn   = torch.nn.MSELoss()

numTotalEpochs = 100000
numEpochs = 10000 # number of epochs to train each iteration
while epoch < numTotalEpochs:
    newCost, newLmbda, newNu = train(network, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    costs.extend(newCost)
    lmbdas.extend(newLmbda)
    nus.extend(newNu)
    epoch += numEpochs

    checkpoint = {'epoch': epoch, 'network': network, 'trainData' : trainData, 'optimiser': optimiser,
                'scheduler': scheduler, 'costs': costs, 'lmbdas' : lmbdas, 'nus' : nus}
    torch.save(checkpoint, 'burgersAdam2.pth') # save network every 'numEpochs' epochs

plotNetwork(network, X, T, XT, u_exact, epoch)
plt.semilogy(costs)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("Cost", fontsize = 16)
plt.title("Burger's Equation Training Cost", fontsize = 16)
plt.show()

plt.plot(lmbdas)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("\u03BB\u2081 Value", fontsize = 16)
plt.title("Burger's Equation \u03BB\u2081 Values", fontsize = 16)
plt.show()

plt.plot(nus)
plt.xlabel("Epochs", fontsize = 16)
plt.ylabel("\u03BB\u2082 Value", fontsize = 16)
plt.title("Burger's Equation \u03BB\u2082 Values", fontsize = 16)
plt.show()

print("Final value of lmbda = ", network.lmbda.item())
print("Final value of nu = ", torch.exp(network.nu).item())
print("True value of lmbda = ", 1.0)
print("True value of nu = ", 0.01 / np.pi)
test(network, XT, u_exact, lossFn)

# for n in network.parameters():
#     print(n)


# %%

# %%
