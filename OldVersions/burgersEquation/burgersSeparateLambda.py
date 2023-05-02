#%%

#############
# Adam optimiser, train for u(x,t) and lambda_i simultaneously
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
        # 3 inputs: x, t, u_exact values
        self.fc1 = torch.nn.Linear(3, numHiddenNodes)
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers-1)])
        # 1 outputs : u
        self.fcLast = torch.nn.Linear(numHiddenNodes, 1)

    def forward(self, input):
        hidden = torch.tanh(self.fc1(input))
        print(hidden.shape)
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
            print(hidden.shape)
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out


def train(network, lambda1, lambda2, lossFn, optimiser, scheduler, loader, numEpochs):
    """Trains the neural network"""
    costList=[]
    lambda1List = []
    lambda2List = []
    network.train(True)
    beta = (1/4)
    for _ in range(numEpochs):
        for batch in loader:

            # print(batch)
            u_out = network.forward(batch)
            # print(u_out)
            du = grad(u_out, batch, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
            # print(du)
            d2u = grad(du, batch, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
            # print(d2u)

            # Use torch.split to preserve grad history
            u_x, u_t, _ = torch.split(du, 1, dim =1)
            # print(u_t)
            # print(u_x)
            u_xx, u_tt, _ = torch.split(d2u, 1, dim =1)
            # print(u_xx)

            _, _, batch_u_exact = torch.split(batch,1, dim =1)

            # DE with exp(lambda2)
            # diffEqLHS = u_t + (lambda1 * u_out * u_x) - (torch.exp(lambda2) * u_xx)

            # DE without exp(lambda2), i.e. just with lambda2
            diffEqLHS = u_t + (lambda1 * u_out * u_x) - (lambda2 * u_xx)

            uLoss = (beta)*lossFn(u_out, batch_u_exact)
            DELoss = (1/beta)* lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            loss = uLoss + DELoss
            loss.backward()

            # Examine lambda1, lambda2 grads
            print("lambda1 grad = ", lambda1.grad)
            print("lambda2 grad = ", lambda2.grad)

            optimiser.step()
            optimiser.zero_grad()

        # update scheduler, tracks loss and updates learning rate if on plateau   
        scheduler.step(loss)

        # store final loss of each epoch
        costList.append(loss.detach().numpy())
        lambda1List.append(lambda1.item())
        lambda2List.append(lambda2.item())

    print("u_train loss = ", uLoss.item())
    print("DE_train loss = ", DELoss.item())
    print("current train loss = ", loss.detach().numpy())

    network.train(False)
    return costList, lambda1List, lambda2List

def test(network, lambda1, lambda2, XT, u_exact, lossFn):
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

    # DE with exp(lambda2)
    # diffEqLHS = u_t + (lambda1 * u_out * u_x) - (torch.exp(lambda2) * u_xx)

    # DE without exp(lambda2), i.e. just with lambda2
    diffEqLHS = u_t + (lambda1 * u_out * u_x) - (lambda2 * u_xx)

    # calculate losses for u, DE, lambda1 and lambda2
    uTestLoss = lossFn(u_out, batch[:,2].view(-1,1))
    DETestLoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))
    lambda1Loss = abs(lambda1 - 1.) * 100
    # lambda2Loss = (abs(torch.exp(lambda2) - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
    lambda2Loss = (abs(lambda2 - ( 0.01 / np.pi)) / (0.01 / np.pi)) * 100
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

    print("lambda1 = ", lambda1.item())
    # print("lambda2 = ", torch.exp(lambda2).item())
    print("lambda2 = ", lambda2.item())


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
# `print(XT.shape)
u_exact = Exact.flatten()[:,None]
# print(u_exact)
# print(u_exact.reshape(X.shape[0],X.shape[1]))
# print(u_exact.shape)

# number of training samples
numSamples = 25600

# try: # load saved network if possible
#     checkpoint = torch.load('burgerAdam.pth')
#     epoch = checkpoint['epoch']
#     network = checkpoint['network']
#     optimiser = checkpoint['optimiser']
#     scheduler = checkpoint['scheduler']
#     losses = checkpoint['losses']
# except: # create new network
epoch = 0
network    = Fitter(numHiddenNodes=16, numHiddenLayers=8)
lambda1 = torch.tensor(torch.rand(1), requires_grad = True) 
lambda2 = torch.tensor(torch.rand(1), requires_grad = True) 
optimiser  = torch.optim.Adam([{"params" : network.parameters()},
                               {"params": lambda1, 'lr' : 1e-3},
                                {"params": lambda2, 'lr' : 1e-3}], 
                                lr = 1e-2)
# optimiser = torch.optim.LBFGS(network.parameters(), lr = 1e-3)
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
losses = []
lambda1List = []
lambda2List = []

trainData = DataSet(XT, u_exact, numSamples)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples/4), shuffle=True)
lossFn   = torch.nn.MSELoss()
# for n in network.parameters():
#     print(n)

iterations = 0
numEpochs = 1000 # number of epochs to train each iteration
while iterations < 100:
    newLoss, newLambda1, newLambda2 = train(network, lambda1, lambda2, lossFn, optimiser, scheduler, trainLoader, numEpochs)
    losses.extend(newLoss)
    lambda1List.extend(newLambda1)
    lambda2List.extend(newLambda2)
    iterations += 1
    epoch += numEpochs

    plotNetwork(network, X, T, XT, u_exact, epoch)

    plt.plot(lambda1List)
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 1")
    plt.title("Lambda 1")
    plt.show()

    plt.plot(lambda2List)
    plt.xlabel("Epochs")
    plt.ylabel("Lambda 2")
    plt.title("Lambda 2")
    plt.show()

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.show()

print("Final value of lambda1 = ", lambda1.item())
# print("Final value of lambda2 = ", torch.exp(lambda2).item())
print("Final value of lambda2 = ", lambda2.item())
print("True value of lambda1 = ", 1.0)
print("True value of lambda2 = ", 0.01 / np.pi)
test(network, lambda1, lambda2, XT, u_exact, lossFn)

# for n in network.parameters():
#     print(n)

# save network
checkpoint = { 
    'epoch': epoch,
    'network': network,
    'optimiser': optimiser,
    'scheduler': scheduler,
    'losses': losses
    }
torch.save(checkpoint, 'burgerAdamSeparate2.pth')

# %%

# %%
