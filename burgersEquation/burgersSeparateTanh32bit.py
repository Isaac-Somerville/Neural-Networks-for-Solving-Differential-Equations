#%%

#############
# Tanh activation function, train for u(x,t), then lambda_i
# lambda_i are not network parameters
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
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, XT, u_exact, numSamples):
        # generate numSamples random indices to get training sample
        idx = np.random.choice(XT.shape[0], numSamples, replace=False) 

        XT_train = torch.tensor(XT[idx,:], requires_grad=True).float()
        u_train = torch.tensor(u_exact[idx,:], requires_grad=True).float()

        # input of forward function must have shape (batch_size, 2)
        # u-values for training must have shape (batch_size, 1)
        # we load this data as a tuple of tensors
        self.data_in = (XT_train, u_train)
        
    def __len__(self):
        return self.data_in[0].shape[0]


    def __getitem__(self, i):
        return (self.data_in[0][i,:], self.data_in[1][i])

class Fitter(torch.nn.Module):
    """Forward propagations"""

    def __init__(self, numHiddenNodes, numHiddenLayers):
        super(Fitter, self).__init__()
        # 2 inputs: x, t
        self.fc1 = torch.nn.Linear(2, numHiddenNodes)
        self.fc1.apply(self.initWeightsXavier)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers)])
        self.fcs.apply(self.initWeightsXavier)
        # 1 output : u
        self.fcLast = torch.nn.Linear(numHiddenNodes, 1)
        self.fcLast.apply(self.initWeightsXavier)

    def forward(self, input):
        input = 2.0*(input - lb)/(ub - lb) - 1.0
        hidden = torch.tanh(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out
    
    def initWeightsXavier(self, layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight, gain=torch.nn.init.calculate_gain('tanh'))  


def trainU(network, lossFn, optimiser, scheduler, loader, numEpochs):
    """
    A function to train a neural network to approximate the solution u(x,t) to Burger's equation
    based on sample data from the exact solution

    Arguments:
    network (Module) -- the neural network
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    scheduler (Learning Rate Scheduler) -- reduces learning rate if cost value is plateauing
    loader (DataLoader) -- generates batches from the training dataset
    numEpochs (int) -- number of training epochs

    Returns:
    costList (list of length 'numEpochs') -- cost values of all epochs
    """
    costList=[]
    network.train(True) # set network into training mode
    for _ in range(numEpochs):
        for batch in loader:
            input, batch_u_exact = batch # separate (x,t) and u(x,t) values
            u_out = network.forward(input) # pass batch of values (x,t) through network

            cost = lossFn(u_out, batch_u_exact) # calculate cost
            cost.backward() # perform back propagation
            optimiser.step() # update parameters
            optimiser.zero_grad() # reset gradients to zero

        scheduler.step(cost) # update scheduler, reduces learning rate if on plateau   
        costList.append(cost.detach().numpy())  # store final cost of each epoch

    network.train(False) # set network out of training mode
    return costList

def trainDE(network, lmbda, nu, lossFn, optimiser, scheduler, loader, numEpochs):
    """
    A function to approximate the parameter values lambda and nu in Burger's equation 
    using a neural network which has been trained to approximate the solution function

    Arguments:
    network (Module) -- the neural network
    lmbda (tensor of shape (1)) -- the parameter lambda
    nu (tensor of shape (1)) -- the parameter nu
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    scheduler (Learning Rate Scheduler) -- reduces learning rate if cost value is plateauing
    loader (DataLoader) -- generates batches from the training dataset
    numEpochs (int) -- number of training epochs

    Returns:
    costList (list of length 'numEpochs') -- cost values of all epochs
    lmbdaList (list of length 'numEpochs') -- lambda values of all epochs
    nuList (list of length 'numEpochs') -- nu values of all epochs
    """
    costList, lmbdaList, nuList = [], [], []
    network.train(True) # set network into train mode
    for batch in loader:
        # calculate u(x,t) and its derivative only once
        input, batch_u_exact = batch # separate (x,t) and u(x,t) values
        u_out = network.forward(input) # pass batch of values (x,t) through network

        # calculate first- and second-order partial derivatives of network output
        du = grad(u_out, input, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
        d2u = grad(du, input, torch.ones_like(du), retain_graph=True, create_graph=True)[0]

        # separate partial derivatives to attain u_t, u_x, u_xx
        # use torch.split to preserve grad history
        u_x, u_t = torch.split(du, 1, dim =1)
        u_xx, u_tt = torch.split(d2u, 1, dim =1)

        for _ in range(numEpochs): # with u and its derivatives fixed, train lambda and nu
            # since we know nu will always be positive, we train with exp(nu)
            diffEqLHS = u_t + (lmbda * u_out * u_x) - (torch.exp(nu) * u_xx)

            cost = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            cost.backward(retain_graph = True) # perform back propagation
            optimiser.step() # update parameters
            optimiser.zero_grad() # reset gradients to zero
            scheduler.step(cost) # update scheduler, reduces learning rate if on plateau   

            # store final cost, lambda- and nu-values of each epoch
            costList.append(cost.item())
            lmbdaList.append(lmbda.item())
            nuList.append(torch.exp(nu).item())

    network.train(False)
    return costList, lmbdaList, nuList

def test(network, lmbda, nu, XT, u_exact, lossFn):
    """
    Tests network solution on all 25600 sample points
    """
    testData = DataSet(XT , u_exact, XT.shape[0])
    input, batch_u_exact = testData.data_in
    u_out = network.forward(input)

    du = grad(u_out, input, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
    # print(du)

    d2u = grad(du, input, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    # print(d2u) 

    u_x, u_t= torch.split(du, 1, dim =1)
    # print(u_t)
    # print(u_x)

    u_xx, u_tt= torch.split(d2u, 1, dim =1)
    # print(u_xx)

    # DE with exp(nu)
    diffEqLHS = u_t + (lmbda * u_out * u_x) - (torch.exp(nu) * u_xx)

    # DE without exp(nu), i.e. just with nu
    # diffEqLHS = u_t + (lmbda * u_out * u_x) - (nu * u_xx)

    # calculate losses for u, DE, lmbda and nu
    uTestLoss = lossFn(u_out, batch_u_exact)
    DETestLoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))
    lmbdaLoss = abs(lmbda - 1.) * 100
    nuLoss = (abs(torch.exp(nu) - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
    # nuLoss = (abs(nu - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
    print("u_test error = ", uTestLoss.item())
    print("DE_test error = ", DETestLoss.item())
    print("lmbda error = ", lmbdaLoss.item(), " %")
    print("nu error = ", nuLoss.item(), " %")
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

    # print("lmbda = ", lmbda.item())
    # print("nu = ", torch.exp(nu).item())
    # print("nu = ", nu.item())


    # print(X.shape)
    # print(T.shape)
    # print(u_out.shape)

    # Plot trial solution
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,T,u_out,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    
    # Plot exact solution
    # ax.scatter(X,T,u_exact, label = 'Exact Solution')
    # ax.legend()

    ax.set_title(str(epoch) + " Epochs")
    plt.show()
    return

# load and format sample data (dictionary) for u(x,t)
# there are 25600 samples in total 
data = scipy.io.loadmat('burgersData.mat')

t = data['t'].flatten()[:,None]
# print(t.shape)
x = data['x'].flatten()[:,None]
# print(x.shape)
Exact = np.real(data['usol']).T
# print(Exact)

X, T = np.meshgrid(x,t)
# print(X.shape)
# print(T.shape)

XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
# print(XT.shape)
u_exact = Exact.flatten()[:,None]
# print(u_exact)
# print(u_exact.reshape(X.shape[0],X.shape[1]))
# print(u_exact.shape)

lb = torch.tensor(XT.min(0)).float()
ub = torch.tensor(XT.max(0)).float()
print(lb)
print(ub)
# XT = 2.0*(XT - lb)/ (ub - lb) - 1.0
    

# number of training samples
numSamples = 2000

try: # load saved network if possible
    checkpoint = torch.load('burgersTanh32Bit.pth')
    epoch = checkpoint['epoch']
    network = checkpoint['network']
    optimiser = checkpoint['optimiser']
    scheduler = checkpoint['scheduler']
    uLosses = checkpoint['uLosses']
    trainData = checkpoint['trainData']
    print("model loaded")
except:
    try: # load initial network
        checkpoint = torch.load('burgersSiluInitialNetwork.pth')
        network = checkpoint['network']
        trainData = checkpoint['trainData']
        print("initial model loaded")
    except: # create new network, save initial conditions
        network   = Fitter(numHiddenNodes=32, numHiddenLayers=8)
        trainData = DataSet(XT, u_exact, numSamples)
        checkpoint = {'network': network,
                       'trainData' : trainData}
        # torch.save(checkpoint, 'burgersSwish10InitialNetwork.pth')
        print("new model created")
    epoch = 0
    optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 
        factor=0.5, 
        patience=500, 
        threshold=1e-4, 
        cooldown=0, 
        min_lr=1e-6, 
        eps=1e-8, 
        verbose=True
    )
    uLosses = []

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples), shuffle=True)
lossFn   = torch.nn.MSELoss()
# for n in network.parameters():
#     print(n)

numEpochs = 10000 # number of epochs to train each iteration
while epoch < 100000:
    newLoss = trainU(network, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    uLosses.extend(newLoss)
    epoch += numEpochs

    plotNetwork(network, X, T, XT, u_exact, epoch)

    plt.semilogy(uLosses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()
    checkpoint = { 
        'epoch': epoch,
        'network': network,
        'optimiser': optimiser,
        'scheduler': scheduler,
        'uLosses': uLosses,
        'trainData': trainData
        }
    torch.save(checkpoint, 'burgersTanh32Bit.pth')
    print("model saved")


lmbda = torch.tensor(torch.rand(1), requires_grad = True) 
nu = torch.tensor(torch.rand(1), requires_grad = True)

# lmbda = torch.tensor(0., requires_grad = True) 
# nu = torch.tensor(0.002479, requires_grad = True)  

plotNetwork(network, X, T, XT, u_exact, epoch)
test(network, lmbda, nu, XT, u_exact, lossFn)

optimiser  = torch.optim.Adam([{"params": lmbda, 'lr' : 1e-3},
                                {"params": nu, 'lr' : 1e-3}], lr = 1e-3)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    factor=0.5, 
    patience=1000, 
    threshold=1e-4, 
    cooldown=0, 
    min_lr=0, 
    eps=1e-8, 
    verbose=True
)

DELosses = []
lmbdaList = []
nuList = []
iterations = 0
numEpochs = 10000 # number of epochs to train each iteration
while iterations < 4:
    newLoss, newLambda1, newLambda2 = trainDE(network, lmbda, nu, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    DELosses.extend(newLoss)
    lmbdaList.extend(newLambda1)
    nuList.extend(newLambda2)
    iterations += 1
    epoch += numEpochs

    plotNetwork(network, X, T, XT, u_exact, epoch)

    plt.plot(lmbdaList)
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 1")
    plt.title("Lambda 1")
    plt.show()

    plt.plot(nuList)
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 2")
    plt.title("Lambda 2")
    plt.show()

    plt.semilogy(DELosses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Differential Equation Loss")
    plt.show()

print("Final value of lmbda = ", lmbda.item())
print("Final value of nu = ", torch.exp(nu).item())
# print("Final value of nu = ", nu.item())
print("True value of lmbda = ", 1.0)
print("True value of nu = ", 0.01 / np.pi)
test(network, lmbda, nu, XT, u_exact, lossFn)
# for n in network.parameters():
#     print(n)

# %%
