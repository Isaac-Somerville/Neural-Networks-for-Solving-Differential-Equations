#%%

#############
# Silu activation function, train for u(x,t) and lambda_i simultaneously
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
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers)])
        # 1 outputs : u
        self.fcLast = torch.nn.Linear(numHiddenNodes, 1)

        # Initialise lambda1, lambda2 as in paper
        # self.lambda1 = torch.nn.Parameter(torch.tensor(0.))
        # self.lambda2 = torch.nn.Parameter(torch.tensor(-6.))

        # Initalise lambda1, lambda2 randomly
        # N.B. should replace exp(lambda2) with lambda2 if doing this
        self.lambda1 = torch.nn.Parameter(torch.rand(1))
        self.lambda2 = torch.nn.Parameter(torch.rand(1))

    def forward(self, input):
        hidden = silu(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = silu(self.fcs[i](hidden))
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out

def silu(x):
    return x * torch.sigmoid(x)

def train(network, lossFn, optimiser, scheduler, loader, numEpochs):
    """Trains the neural network"""
    costList = []
    lambda1List = []
    lambda2List = []
    network.train(True)
    for _ in range(numEpochs):
        for batch in loader:
            input, batch_u_exact = batch
            u_out = network.forward(input)
            # print(u_out)
            du = grad(u_out, input, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
            # print(du)
            d2u = grad(du, input, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
            # print(d2u)

            # Use torch.split to preserve grad history
            u_x, u_t = torch.split(du, 1, dim =1)
            # print(u_t)
            # print(u_x)
            u_xx, u_tt = torch.split(d2u, 1, dim =1)
            # print(u_xx)

            # DE with exp(lambda2)
            diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)

            # DE without exp(lambda2), i.e. just with lambda2
            # diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (network.lambda2 * u_xx)

            uLoss = lossFn(u_out, batch_u_exact)
            DELoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            loss = uLoss + DELoss
            loss.backward()

            # Examine lambda1, lambda2 grads
            # print("lambda1 grad = ", network.lambda1.grad)
            # print("lambda2 grad = ", network.lambda2.grad)

            optimiser.step()
            optimiser.zero_grad()

        # update scheduler, tracks loss and updates learning rate if on plateau   
        scheduler.step(loss)

        # store final loss of each epoch
        costList.append(loss.detach().numpy())
        lambda1List.append(network.lambda1.item())
        lambda2List.append(torch.exp(network.lambda2).item())

    print("u_train loss = ", uLoss.item())
    print("DE_train loss = ", DELoss.item())
    print("current train loss = ", loss.detach().numpy())

    network.train(False)
    return costList, lambda1List, lambda2List

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

    diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)

    # calculate losses for u, DE, lambda1 and lambda2
    uTestLoss = lossFn(u_out, batch_u_exact)
    DETestLoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))
    lambda1Loss = abs(network.lambda1 - 1.) * 100
    lambda2Loss = (abs(torch.exp(network.lambda2) - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
    print("u_test error = ", uTestLoss.item())
    print("DE_test error = ", DETestLoss.item())
    print("lambda1 error = ", lambda1Loss.item(), " %")
    print("lambda2 error = ", lambda2Loss.item(), " %")
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
    lambda1 = network.lambda1
    lambda2 = network.lambda2
    print("lambda1 = ", lambda1.item())
    print("lambda2 = ", torch.exp(lambda2).item())

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

# load and format sample data (dictionary) for u(x,t)
# there are 25600 samples in total 
data = scipy.io.loadmat('burgersData.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T
# print(Exact)

X, T = np.meshgrid(x,t)

XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
# `print(XT.shape)
u_exact = Exact.flatten()[:,None]
# print(u_exact)
# print(u_exact.reshape(X.shape[0],X.shape[1]))
# print(u_exact.shape)

# number of training samples
numSamples = 2000

try: # load saved network if possible
    checkpoint = torch.load('burgersSimultaneousSilu32bit.pth')
    epoch = checkpoint['epoch']
    network = checkpoint['network']
    trainData = checkpoint['trainData']
    optimiser = checkpoint['optimiser']
    scheduler = checkpoint['scheduler']
    losses = checkpoint['losses']
    lambda1s = checkpoint['lambda1s']
    lambda2s = checkpoint['lambda2s']
    print("model loaded")
except: # create new network
    epoch = 0
    network    = Fitter(numHiddenNodes=32, numHiddenLayers=8)
    optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-3)
    # optimiser = torch.optim.LBFGS(network.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 
        factor=0.5, 
        patience=500, 
        threshold=1e-4, 
        cooldown=0, 
        min_lr=0, 
        eps=1e-8, 
        verbose=True
    )
    losses = []
    lambda1s = []
    lambda2s = []
    trainData = DataSet(XT, u_exact, numSamples)
    print("new model created")

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=numSamples, shuffle=True)
lossFn   = torch.nn.MSELoss()
# for n in network.parameters():
#     print(n)

iterations = 0
numEpochs = 10000 # number of epochs to train each iteration
while iterations < 5:
    newLoss, newLambda1, newLambda2 = train(network, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    losses.extend(newLoss)
    lambda1s.extend(newLambda1)
    lambda2s.extend(newLambda2)
    iterations += 1
    epoch += numEpochs

    plotNetwork(network, X, T, XT, u_exact, epoch)
    plt.semilogy(losses)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("Cost", fontsize = 16)
    plt.title("Burger's Equation Training Cost", fontsize = 16)
    plt.show()

    plt.plot(lambda1s)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("\u03BB\u2081 Value", fontsize = 16)
    plt.title("Burger's Equation \u03BB\u2081 Values", fontsize = 16)
    plt.show()

    plt.plot(lambda2s)
    plt.xlabel("Epochs", fontsize = 16)
    plt.ylabel("\u03BB\u2082 Value", fontsize = 16)
    plt.title("Burger's Equation \u03BB\u2082 Values", fontsize = 16)
    plt.show()
    
    # save network
    checkpoint = { 
    'epoch': epoch,
    'network': network,
    'trainData' : trainData,
    'optimiser': optimiser,
    'scheduler': scheduler,
    'losses': losses,
    'lambda1s' : lambda1s,
    'lambda2s' : lambda2s
    }
    torch.save(checkpoint, 'burgersSimultaneousSilu32bit.pth')

print("Final value of lambda1 = ", network.lambda1.item())
print("Final value of lambda2 = ", torch.exp(network.lambda2).item())
print("True value of lambda1 = ", 1.0)
print("True value of lambda2 = ", 0.01 / np.pi)
test(network, XT, u_exact, lossFn)

# for n in network.parameters():
#     print(n)


# %%

# %%
