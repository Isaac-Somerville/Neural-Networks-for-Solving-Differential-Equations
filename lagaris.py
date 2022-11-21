#%%
"""""
Solves problem 2 from lagaris
Notes:
Switch from SGD to Adam
Extrapolates VERY POORLY
LR = 1e-3, converges after 6000 epochs, loss 2e-4
LR = 1e-1, converges after 6000 epochs, loss 1e-5
10 hidden nodes, conv after 6000 epochs, loss = 1e-3
50 hidden nodes, conv after 6000 epochs, loss = 2e-4

HW: read Nielsen
Look into local minima stuff
Generalise this code
"""

import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt

class DataSet(torch.utils.data.Dataset):
    def __init__(self, num_samples, xrange):
        self.data_in  = torch.linspace(xrange[0], xrange[1], num_samples, requires_grad=True)
        
        # x = torch.linspace(xrange[0], xrange[1]/2, int(num_samples/4), requires_grad=True)
        # y = torch.linspace(xrange[1]/2, xrange[1], int(3 * num_samples/4), requires_grad=True)
        # self.data_in = torch.cat((x,y),0)

    def __len__(self):
        return len(self.data_in)

    def __getitem__(self, i):
        return self.data_in[i]

class Fitter(torch.nn.Module):
    def __init__(self, num_hidden_nodes):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, x):
        hidden = torch.sigmoid(self.fc1(x))
        y = self.fc2(hidden)
        return y

def train(network, loader, loss_fn, optimiser, epochs, iterations):
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            batch = batch.view(-1, 1)
            n_out = network(batch)
            # Get the derivative of the network output with respect
            # to the input values. 
            dndx = torch.autograd.grad(n_out, batch, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            # phi(x) = x * N(x), or in code:
            phit = batch*n_out
            # phi'(x) = N(x) + x N'(x), or in code;
            dphidx = n_out + batch*dndx

            x=batch.detach().numpy()
            f = torch.tensor(np.exp(-x/5)*np.cos(x)) - (1/5)*phit
            diffeq = dphidx - f
            loss = loss_fn(diffeq, torch.zeros_like(diffeq))
            cost_list.append(loss.detach().numpy())
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
        if epoch%(epochs/5)==0:
            plotit(network, epoch, epochs, iterations)
    network.train(False)
    return cost_list



def plotit(network, epoch, epochs, iterations):
    x    = torch.Tensor(np.linspace(xrange[0], xrange[1], num_samples)).view(-1,1)
    
    # Focus datapoints on right half of xrange where graph changes more
    # x1 = torch.linspace(xrange[0], xrange[1]/2, int(num_samples/4)).view(-1,1)
    # x2 = torch.linspace(xrange[1]/2, xrange[1], int(3 * num_samples/4)).view(-1,1)
    # x = torch.cat((x1,x2),0 )
    
    N     = network.forward(x).detach().numpy()
    exact = np.exp(-x/5)*np.sin(x)
    plt.plot(x, x*N, 'r-', label = "Neural Network Output")
    plt.plot(x, exact, 'b.', label = "True Solution")
    
    # Extrapolate to left and right of training data
    # x2     = torch.Tensor(np.linspace(xrange[0]-5, xrange[0], int(num_samples/2))).view(-1,1)
    # x3 = torch.Tensor(np.linspace(xrange[1], xrange[1] + 5, int(num_samples/2))).view(-1,1)
    # N2     = network.forward(x2).detach().numpy()
    # N3     = network.forward(x3).detach().numpy()
    # exact2 = np.exp(-x2/5)*np.sin(x2)
    # exact3 = np.exp(-x3/5)*np.sin(x3)
    # plt.plot(x2, x2*N2, 'r-')
    # plt.plot(x3, x3*N3, 'r-') 
    # plt.plot(x2, exact2, 'g.', label = "True Solution")
    # plt.plot(x3, exact3, 'g.')
    
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend(loc = "upper right")
    plt.ylim([-0.5, 1])
    # plt.ylim([-1.5, 2.7])
    plt.title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()
    
xrange=[0, 10]
num_samples = 30
network      = Fitter(num_hidden_nodes=10)
train_set    = DataSet(num_samples,  xrange)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=60, shuffle=True)
loss_fn      = torch.nn.MSELoss()
# This does not work *at all* if you use SGD as the optimiser. It just does
# not manage to get the oscillations right. For Adam it is miraculously good.
optimiser    = torch.optim.Adam(network.parameters(), lr=1e-3)
# optimiser    = torch.optim.SGD(network.parameters(), lr=1e-3)


#train(network, train_loader, loss_fn, optimiser, 2000)

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.001 and iterations < 10:
    losses.extend( train(network, train_loader, loss_fn, optimiser, epochs, iterations) )
    iterations += 1
losses = losses[1:]
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")
# %%
