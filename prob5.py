#%%
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt


class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, xrange, yrange, num_samples):
        # self.data_in  = torch.rand(num_samples, requires_grad=True)
        X  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
        Y  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
        data = [[[x,y] for y in Y] for x in X]
        tensor_data = torch.tensor(data, requires_grad=True)
        self.data_in = tensor_data.view(-1,2)
        # self.data_in = torch.meshgrid(X,Y)
        # print(self.data_in.shape)

    def __len__(self):
        return len(self.data_in)

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

    def trial_term(self,x,y):
        """
        First term in trial solution that helps to satisfy BCs
        """
        e_inv = np.exp(-1)
        return ((1-x)*y**3 + x*(1+y**3)*e_inv + (1-y)*x*(torch.exp(-x)-e_inv) +
                y * ((1+x)*torch.exp(-x) - (1-x-2*x*e_inv)))
    
    def trial(self,x,y,n_out):
        return self.trial_term(x,y) + x*(1-x)*y*(1-y)*n_out
    
    def diffEq(self,x,y,trial):
        trial_dx = torch.autograd.grad(trial, x, torch.ones_like(trial), retain_graph=True, create_graph=True)[0]
        trial_dx2 = torch.autograd.grad(trial_dx, x, torch.ones_like(trial_dx), retain_graph=True, create_graph=True)[0]
        trial_dy = torch.autograd.grad(trial, y, torch.ones_like(trial), retain_graph=True, create_graph=True)[0]
        trial_dy2 = torch.autograd.grad(trial_dy, y, torch.ones_like(trial_dy), retain_graph=True, create_graph=True)[0]
        RHS = torch.exp(-x) * (x - 2 + y**3 + 6*y)
        return trial_dx2 + trial_dy2 - RHS


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            n_out = network(batch)
            input = batch.transpose(0,1)
            x, y = input[0].view(-1,1), input[1].view(-1,1)

            # Get value of trial solution f(x)
            trial = diffEq.trial(x,y,n_out)
    
            # Get value of diff equations D(x) = 0
            D = diffEq.diffEq(x, y, trial)

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
    data = [[[x,y] for y in Y] for x in X]
    input = torch.tensor(data, requires_grad=True).view(-1,2)
    print(input)
    N = network.forward(input)
    print(N)
    N = N.detach().numpy().reshape(10,10)
    print(N)

    x , y = torch.meshgrid(X,Y)
    print(x)
    print(y)
    exact = diffEq.solution(x,y).detach().numpy()
    x = x.detach().numpy()
    y = y.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x,y,N,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(x,y,exact)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend(loc = "lower center")
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    ax.view_init(30, 315)
    plt.show()


num_samples = 10
xrange = [0,1]
yrange = [0,1]
diffEq = DiffEq(xrange, yrange, num_samples)

network     = Fitter(num_hidden_nodes=10)
loss_fn      = torch.nn.MSELoss()
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-2)
train_set    = DataSet(xrange,yrange,num_samples)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=10, shuffle=True)

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.0001  and iterations < 10:
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