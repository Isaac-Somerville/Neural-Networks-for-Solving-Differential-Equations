#%%
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
        idx = np.random.choice(XT.shape[0], numSamples, replace=False)
        XT_train = torch.tensor(XT[idx,:], requires_grad=True)
        u_train = torch.tensor(u_exact[idx,:], requires_grad=True)

        # input of forward function must have shape (batch_size, 3)
        self.data_in = torch.cat((XT_train,u_train),1)
        
    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, i):
        return self.data_in[i]

class Fitter(torch.nn.Module):
    """Forward propagations"""

    def __init__(self, numHiddenNodes, numHiddenLayers):
        super(Fitter, self).__init__()
        # 3 inputs: x, t, u values
        self.fc1 = torch.nn.Linear(3, numHiddenNodes)
        self.fcs = [
            torch.nn.Linear(numHiddenNodes, numHiddenNodes)
            for _ in range(numHiddenLayers)
        ]
        
        # 3 outputs : u, lambda1, lambda2
        self.fcLast = torch.nn.Linear(numHiddenNodes, 3)

    def forward(self, input):
        hidden = torch.tanh(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out

def diffEq(u, u_t, u_x, u_xx, lambda1, lambda2):
    """
    Returns the LHS of Burger's Equation
    """
    return u_t + (lambda1 * u * u_x) - (lambda2 * u_xx)

def train(network, lossFn, optimiser, loader, epochs, iterations):
    """Trains the neural network"""
    costList=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            n_out = network.forward(batch)
            u, lambda1, lambda2 = torch.split(n_out, 1, dim=1)
            print(u)
            du = grad(u, batch, retain_graph=True, create_graph=True)[0]
            print(du)
            d2u = grad(du, batch, retain_graph=True, create_graph=True)[0]
            print(d2u)
            u_t = du[:,1].view(-1,1)
            print(u_t)
            u_x = du[:,0].view(-1,1)
            print(u_x)
            u_xx = d2u[:,0].view(-1,1)
            print(u_xx)

            diffEqLHS = diffEq(u, u_t, u_x, u_xx, lambda1, lambda2)

            u_exact = batch[:,2].view(-1,1)
            print(u_exact)

            uLoss = lossFn(u, u_exact)
            DELoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            loss = uLoss + DELoss

            loss.backward()
            optimiser.step()

            #store final loss of each epoch
            costList.append(loss.detach().numpy())

            if epoch == epochs:
                plotNetwork()
                print("current loss = ", loss.detach().numpy())

    network.train(False)
    return costList


def plotNetwork(network, XT, u_exact, epoch, epochs, iterations):
    



data = scipy.io.loadmat('burgersData.mat')

t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_exact = Exact.flatten()[:,None]

numSamples = 2000
trainData = DataSet(XT, u_exact, numSamples)