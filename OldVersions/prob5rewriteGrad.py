#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

# maybe jacobian instead?


class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, xrange, yrange, num_samples):
        # self.data_in  = torch.rand(num_samples, requires_grad=True)
        X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
        Y  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
        grid = torch.meshgrid(X,Y)
        # print(grid)
        # print(grid[0].reshape(-1,1))
        # print(grid[1].reshape(-1,1))
        self.data_in = torch.cat((grid[0].reshape(-1,1),grid[1].reshape(-1,1)),1)
        #print(self.data_in.shape)
        #tensor_data = torch.tensor(data, requires_grad=True)
        #self.data_in = tensor_data.view(-1,2)
        # self.data_in = torch.meshgrid(X,Y)
        # print(self.data_in.shape)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, i):
        return self.data_in[i]

class Fitter(torch.nn.Module):
    """Forward propagations"""
    def __init__(self, num_hidden_nodes):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(2, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, input):
        hidden = torch.sigmoid(self.fc1(input))
        z = self.fc2(hidden)
        return z

class DiffEq():
    """
    Differential equation from Lagaris et al. problem 5
    This problem is a PDE in two variables, with Dirichlet
    boundary conditions
    """
    def __init__(self, xrange, yrange, num_samples):
        self.xrange = xrange
        self.yrange = yrange
        self.num_samples = num_samples

    def solution(self,x,y):
        return torch.exp(-x) * (x + y**3)

    def trial_term(self,batch):
        """
        First term in trial solution that helps to satisfy BCs
        """
        e_inv = np.exp(-1)
        return ((1-batch[:,0])*(batch[:,1]**3) + batch[:,0]*(1+(batch[:,1]**3))*e_inv + (1-batch[:,1])*batch[:,0]*(torch.exp(-batch[:,0])-e_inv) +
                batch[:,1] * ((1+batch[:,0])*torch.exp(-batch[:,0]) - (1-batch[:,0]-(2*batch[:,0]*e_inv))))
    
    def trial(self,batch,n_out):
        return (self.trial_term(batch) + batch[:,0]*(1-batch[:,0])*batch[:,1]*(1-batch[:,1])*n_out
    ).view(-1,1)

    def diffEq(self,batch,trial):
        #print(trial)
        d_trial = grad(trial, batch, torch.ones_like(trial), retain_graph=True, create_graph=True)[0]
        d2_trial = grad(d_trial, batch, torch.ones_like(d_trial), retain_graph=True, create_graph=True)[0]
        # print(d_trial)
        # print(d2_trial)
        dx2_trial, dy2_trial = d2_trial[:,0], d2_trial[:,1]
        # print(dx2_trial)
        # print(dy2_trial)
        RHS = torch.exp(-batch[:,0]) * (batch[:,0] - 2 + batch[:,1]**3 + 6*batch[:,1])
        return dx2_trial + dy2_trial - RHS


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            #print(batch)
            n = network(batch)
            #print(n)
            n_out = n.view(1,-1)
            #print(n_out)
            #print(x)
            #print(y)

            # Get value of trial solution f(x)
            trial = diffEq.trial(batch,n_out)
    
            # Get value of diff equations D(x) = 0
            D = diffEq.diffEq(batch, trial)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))
            cost_list.append(loss.detach().numpy())
        
            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            
        if epoch%(epochs/5)==0:
        #if epoch == epochs:
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange)
        
    network.train(False)
    return cost_list

def plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
    Y  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
    x,y = torch.meshgrid(X,Y)
    input = torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)
    # print(x.reshape(-1,1))
    # print(y.reshape(-1,1))
    #data = [[[x,y] for y in Y] for x in X]
    #input = torch.tensor(data, requires_grad=True).view(-1,2)
    #print(input)
    N = network.forward(input)
    # print(N)
    N = N.reshape(num_samples,num_samples).detach().numpy()
    # print(x)
    # print(y)s
    # print(N)

    exact = diffEq.solution(x,y).detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x,y,N,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(x,y,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    #ax.view_init(10, 270)
    plt.show()

network     = Fitter(num_hidden_nodes=16)
loss_fn      = torch.nn.MSELoss()
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-2)

# j = 4
# ranges = []
# for i in range(j):
#     lst = [i/j, (i+1)/j]
#     ranges.append(lst)
# ranges.append([0,1])
ranges = [[0,1]]
for xrange in ranges:
    yrange = xrange
    num_samples = 5
    diffEq = DiffEq(xrange, yrange, num_samples)
    train_set    = DataSet(xrange,yrange,num_samples)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=5, shuffle=True)  

    losses = [1]
    iterations = 0
    epochs = 5000
    while losses[-1] > 0.001  and iterations < 10:
        newLoss = train(network, train_loader, loss_fn,
                            optimiser, diffEq, epochs, iterations)
        losses.extend(newLoss)
        iterations += 1
    losses = losses[1:]
    print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()

plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])

#%%






# # Plot for a three-dimensional surface
# x = np.linspace(0,1,20)
# y = np.linspace(0,1,20)
# X,Y = np.meshgrid(x,y)
# Z = np.exp(-X) * (X+Y**3)
# ax = plt.axes(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
#                 cmap='plasma', edgecolor='none')

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
# ax.view_init(30, 315)


# Data for three-dimensional scattered points
# zdata = 15 * np.random.random(100)
# xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
# ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
# ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')