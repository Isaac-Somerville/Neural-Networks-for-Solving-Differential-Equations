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

        self.lambda1 = torch.nn.Parameter(torch.tensor(1.))
        self.lambda2 = torch.nn.Parameter(torch.tensor(-6.))
        # self.lambda1 = torch.nn.Parameter(torch.rand(1))
        # self.lambda2 = torch.nn.Parameter(torch.rand(1))

    def forward(self, input):
        hidden = torch.tanh(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        # No activation function on final layer
        out = self.fcLast(hidden)
        return out


def train(network, lossFn, optimiser, scheduler, loader, epochs, 
            iterations, X, T, XT, u_exact):
    """Trains the neural network"""
    costList=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:

            ######### WITH LBFGS ALGORITHM
            # def closure():
            #     print(epoch)
            #     optimiser.zero_grad()
            #     u_out = network.forward(batch)

            #     du = grad(u_out, batch, torch.ones_like(u_out), retain_graph=True, create_graph=True)[0]
            #     # print(du)
            #     d2u = grad(du, batch, torch.ones_like(du), retain_graph=True, create_graph=True)[0]
            #     # print(d2u)
            #     u_x, u_t, _ = torch.split(du, 1, dim =1)
            #     # print(u_t)
            #     # print(u_x)
            #     u_xx, u_tt, _ = torch.split(d2u, 1, dim =1)
            #     # print(u_xx)

            #     # print(network.lambda1)
            #     # print(network.lambda2)

            #     # diffEqLHS = diffEq(u_out, u_t, u_x, u_xx, network.lambda1, network.lambda2)
            #     diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)

            #     # batch_u_exact = batch[:,2].view(-1,1)
            #     _, _, batch_u_exact = torch.split(batch,1, dim =1)
            #     # print(u_exact)

            #     uLoss = lossFn(u_out, batch_u_exact)
            #     DELoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            #     loss = uLoss + DELoss

            #     loss.backward()

            #     costList.append(loss.item())
            #     if epoch == epochs:
            #         # test(network, XT, u_exact, lossFn)
            #         plotNetwork(network, X, T, XT, u_exact, epoch, epochs, iterations)
            #         print("u_train loss = ", uLoss.item())
            #         print("DE_train loss = ", DELoss.item())
            #         print("current train loss = ", loss.detach().numpy())
            #     return loss

            # optimiser.step(closure)

            ######### 

            ######### WITH ADAM
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
 
            # print(network.lambda1)
            # print(network.lambda2)

            _, _, batch_u_exact = torch.split(batch,1, dim =1)
            # print(u_exact)

            # diffEqLHS = diffEq(u_out, u_t, u_x, u_xx, network.lambda1, network.lambda2)
            diffEqLHS = u_t + (network.lambda1 * u_out * u_x) - (torch.exp(network.lambda2) * u_xx)

            uLoss = lossFn(u_out, batch_u_exact)
            DELoss = lossFn(diffEqLHS, torch.zeros_like(diffEqLHS))

            loss = uLoss + DELoss
            loss.backward()
            print("lambda1 grad = ", network.lambda1.grad)
            print("lambda2 grad = ", network.lambda2.grad)
            optimiser.step()
            optimiser.zero_grad()

        # update scheduler, tracks loss and updates learning rate if on plateau   
        scheduler.step(loss)

        # store final loss of each epoch
        costList.append(loss.detach().numpy())

        if epoch == epochs:
            # test(network, XT, u_exact, lossFn)
            plotNetwork(network, X, T, XT, u_exact, epoch, epochs, iterations)
            print("u_train loss = ", uLoss.item())
            print("DE_train loss = ", DELoss.item())
            print("current train loss = ", loss.detach().numpy())

        ###########

    network.train(False)
    return costList

def test(network, XT, u_exact, lossFn):
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

    # print(network.lambda1)
    # print(network.lambda2)

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


def plotNetwork(network, X, T, XT, u_exact, epoch, epochs, iterations):
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
    lambda1 = network.lambda1
    lambda2 = network.lambda2
    print("lambda1 = ", lambda1.item())
    print("lambda2 = ", torch.exp(lambda2).item())

    # print(X.shape)
    # print(T.shape)
    # print(u_out.shape)

    # Plot both trial and exact solutions
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,T,u_out,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    # ax.scatter(X,T,u_exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    # ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    #ax.view_init(30, 315)
    plt.show()
    return


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

numSamples = 1000
# network    = Fitter(numHiddenNodes=32, numHiddenLayers=16)
network = torch.load('burger.pt')
trainData = DataSet(XT, u_exact, numSamples)
trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=int(numSamples), shuffle=True)
lossFn   = torch.nn.MSELoss()
for n in network.parameters():
    print(n)
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-2)
# optimiser = torch.optim.LBFGS(network.parameters(), lr = 1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    factor=0.1, 
    patience=500, 
    threshold=1e-4, 
    cooldown=0, 
    min_lr=0, 
    eps=1e-8, 
    verbose=True
    )

losses = [1]
iterations = 0
epochs = 1000
while iterations < 5:
    newLoss = train(network, lossFn, optimiser, scheduler, trainLoader, 
                    epochs, iterations, X, T, XT, u_exact)
    losses.extend(newLoss)
    iterations += 1
    plt.semilogy(losses[1:])
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()
losses = losses[1:]
# print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")
# plt.semilogy(losses)
# plt.xlabel("Epochs")
# plt.ylabel("Log of Loss")
# plt.title("Loss")
# plt.show()

print("True value of lambda1 = ", 1.0)
print("True value of lambda2 = ", 0.01 / np.pi)
plotNetwork(network, X, T, XT, u_exact, 0, epochs, iterations)

for n in network.parameters():
    print(n)

# torch.save(network, 'burger.pt')

# %%