#%%

#############
# Adam optimiser, trains for u(x,t) first, then lambda_i
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

        # input of forward function must have shape (batch_size, 3)
        self.data_in = torch.cat((XT_train,u_train),1)
        
    def __len__(self):
        return self.data_in.shape[0]


    def __getitem__(self, i):
        return self.data_in[i,:]

class Fitter(torch.nn.Module):
    """Forward propagations"""

    def __init__(self, numHiddenNodes, numHiddenLayers):
        super(Fitter, self).__init__()
        # 3 inputs: x, t, u values
        self.fc1 = torch.nn.Linear(3, numHiddenNodes)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers-1)])
        # 1 outputs : u
        self.fcLast = torch.nn.Linear(numHiddenNodes, 1)

        # Initialise lambda1, lambda2 as in paper
        self.lambda1 = torch.nn.Parameter(torch.tensor(0.))
        self.lambda2 = torch.nn.Parameter(torch.tensor(-6.))

        # Initalise lambda1, lambda2 randomly
        # N.B. should replace exp(lambda2) with lambda2 if doing this
        # self.lambda1 = torch.nn.Parameter(torch.rand(1))
        # self.lambda2 = torch.nn.Parameter(torch.rand(1))

    def forward(self, input):
        hidden = torch.tanh(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out


def trainU(network, lossFn, optimiser, scheduler, loader, numEpochs):
    """Trains the neural network to approximate u(x,t)"""
    costList=[]
    network.train(True)
    for _ in range(numEpochs):
        for batch in loader:
            # print(batch)
            u_out = network.forward(batch)
            # print(u_out)

            _, _, batch_u_exact = torch.split(batch,1, dim =1)
            # print(u_exact)

            loss = lossFn(u_out, batch_u_exact)

            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        # update scheduler, tracks loss and updates learning rate if on plateau   
        scheduler.step(loss)

        # store final loss of each epoch
        costList.append(loss.detach().numpy())

    print("current u train loss = ", loss.detach().numpy())     
    network.train(False)
    return costList

def trainDE(network, lossFn, optimiser, scheduler, loader, numEpochs):
    """Trains the neural network to approximate lambda1, lambda2"""
    costList=[]
    network.train(True)
    for _ in range(numEpochs):
        for batch in loader:
            # print(batch)
            u_out = network.forward(batch)
            # print(u_out)
            du = grad(u_out, batch, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
            # print(du)
            d2u = grad(du, batch, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
            # print(d2u)
            u_x, u_t, _ = torch.split(du, 1, dim =1)
            # print(u_t)
            # print(u_x)
            u_xx, u_tt, _ = torch.split(d2u, 1, dim =1)
            # print(u_xx)
            
            # With exponential for lambda2
            diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)

            # Without exponential for lambda2
            # diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (network.lambda2 * u_xx)

            loss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            loss.backward()

            # Examine grads on lambda1, lambda2
            print("lambda1 grad = ", network.lambda1.grad.item())
            print("lambda2 grad = ", network.lambda2.grad.item())
            optimiser.step()
            optimiser.zero_grad()

        # update scheduler, tracks loss and updates learning rate if on plateau   
        scheduler.step(loss)

        # store final loss of each epoch
        costList.append(loss.detach().numpy())

        

    print("current DE train loss = ", loss.detach().numpy())
    
    print("lambda1 = ", network.lambda1.item())
    # With exponential for lambda2
    print("lambda2 = ", torch.exp(network.lambda2))

    # Without exponential for lambda2
    # print("lambda2 = ", network.lambda2.item())
    print("lambda1 grad = ", network.lambda1.grad.item())
    print("lambda2 grad = ", network.lambda2.grad.item())

    network.train(False)
    return costList

def test(network, XT, u_exact, lossFn):
    """
    Tests network solution on all 25600 sample points
    """
    testData = DataSet(XT , u_exact, XT.shape[0])
    batch = testData.data_in
    u_out = network.forward(batch)
    du = grad(u_out, batch, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
    # print(du)
    d2u = grad(du, batch, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
    # print(d2u)
    u_x, u_t, _ = torch.split(du, 1, dim =1)
    # print(u_t)
    # print(u_x)
    u_xx, u_tt, _ = torch.split(d2u, 1, dim =1)
    # print(u_xx)

    diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)
    uTestLoss = lossFn(u_out, batch[:,2].view(-1,1))
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

    input = torch.cat((XT,u_exact),1)
    # print(X)
    # print(T)
    u_out = network.forward(input)
    # print(u_out)
    u_out = u_out.reshape(X.shape[0],X.shape[1])
    # print(u_out)
    u_out = u_out.detach().numpy()

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
x = data['x'].flatten()[:,None]
Exact = np.real(data['usol']).T
# print(Exact)

X, T = np.meshgrid(x,t)

XT = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
# print(XT.shape)
u_exact = Exact.flatten()[:,None]
# print(u_exact)
# print(u_exact.reshape(X.shape[0],X.shape[1]))
# print(u_exact.shape)

# number of training samples
numSamples = 2000
epoch = 0 
network    = Fitter(numHiddenNodes=16, numHiddenLayers=8)
trainData = DataSet(XT, u_exact, numSamples)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=numSamples, shuffle=True)
lossFn   = torch.nn.MSELoss()

# for n in network.parameters():
#     print(n)

optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-3)
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

# Don't update lambda_i while training u
network.lambda1.requires_grad = False
network.lambda2.requires_grad = False

uLosses = []
iterations = 0
numEpochs = 1000 # number of epochs to train each iteration
while iterations < 5:
    newLoss = trainU(network, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    uLosses.extend(newLoss)
    iterations += 1
    epoch += numEpochs
    plt.semilogy(uLosses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("u Loss")
    plt.show()
    plotNetwork(network, X, T, XT, u_exact, epoch)

test(network, XT, u_exact, lossFn)

# stop updating weights and biases
for n in network.parameters():
    n.requires_grad = False

# reset learning rate (in case scheduler altered it)
for g in optimiser.param_groups:
        g['lr'] = 1e-2

# update lambda1, lambda2
network.lambda1.requires_grad = True
network.lambda2.requires_grad = True

DELosses = []
iterations = 0
while iterations < 5:
    newLoss = trainDE(network, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    DELosses.extend(newLoss)
    iterations += 1
    plt.semilogy(DELosses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("DE Loss")
    plt.show()

print("Final value of lambda1 = ", network.lambda1.item())
print("Final value of lambda2 = ", torch.exp(network.lambda2).item())
print("True value of lambda1 = ", 1.0)
print("True value of lambda2 = ", 0.01 / np.pi)
test(network, XT, u_exact, lossFn)

# for n in network.parameters():
#     print(n)

# %%
