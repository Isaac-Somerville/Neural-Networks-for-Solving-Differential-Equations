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
        grid = torch.meshgrid(X,Y)
        self.data_in = torch.cat((grid[0].reshape(-1,1),grid[1].reshape(-1,1)),1)
        # print(self.data_in)

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
    Differential equation from Lagaris et al. problem 7
    This problem is a PDE in two variables, with mixed
    boundary conditions
    """
    def __init__(self, xrange, yrange, num_samples):
        self.xrange = xrange
        self.yrange = yrange
        self.num_samples = num_samples

    def solution(self, x, y):
        return (y**2) * torch.sin(np.pi * x)

    def trial_term(self,x,y):
        """
        First term B(x,y) in trial solution that helps to satisfy BCs
        f(0,y) = 0, f(1,y) = 0, f(x,0) = 0, f_{y}(x,1) = 2*sin(pi*x)
        B(x,y) = y * 2 * sin(pi*x)
        """
        return 2 * y * torch.sin(np.pi * x)

    def trial(self,x,y,n_xy,n_x1,dndy_x1):
        return self.trial_term(x,y) + x*(1-x)*y*(n_xy - n_x1 - dndy_x1)

    def dx_trial(self,x,y,n_xy, n_x1, dndy_x1, dndx, dndx_x1, dndydx_x1):
        """
        df / dx = 2*y*pi*cos(pi*x) + 
                    y * [(1-2*x)(N - N(x,1) - N_{y}(x,1)) 
                        + x(1-x)(N_{x} - N_{x}(x,1) - N_{xy}(x,1)]
        """
        return ( 2* y *np.pi * torch.cos(np.pi*x) 
                + y * ((1-2*x) * (n_xy - n_x1 - dndy_x1)
                    + x*(1-x)*(dndx - dndx_x1 - dndydx_x1)))

    def dx2_trial(self, x,y,n_xy, n_x1, dndy_x1, dndx, dndx_x1, dndydx_x1, dndx2, dndx2_x1, dndydx2_x1):
        """
        d^2f / dx^2 = -2*y*pi^2*sin(pi*x) 
                    + y * [ (-2)*(N - N(x,1) - N_{y}(x,1)
                            + 2(1-2x)((N_{x} - N_{x}(x,1) - N_{xy}(x,1))
                            + x(1-x)(N_{xx} - N_{xx}(x,1) - N_{xxy}(x,1))]"""
        return ( -2* y *(np.pi)**2 * torch.sin(np.pi*x) 
                + y * ( (-2) * (n_xy - n_x1 - dndy_x1)
                    + 2*(1-2*x) * (dndx - dndx_x1 - dndydx_x1)
                    + x*(1-x)*(dndx2 - dndx2_x1 - dndydx2_x1)))

    def dy_trial(self,x,y, n_xy, n_x1, dndy_x1, dndy):
        """
        df / dy = 2sin(pi*x) + x(1-x)[(N - N(x,1) - N_{y}(x,1))
                                    + y * N_{y}]
        """
        return (2*torch.sin(np.pi *x) + x*(1-x) *
            ((n_xy - n_x1 - dndy_x1) + (y* dndy)))

    def dy2_trial(self,x,y,dndy,dndy2):
        """
        d^2f / dy^2 = x(1-x)[2N_{y} + y * N_{yy}]
        """
        return (x * (1-x) * (2 * dndy + y * dndy2))

    def diffEq(self,x,y,trial_dx2,trial_dy2):
        RHS = (2-((np.pi*y)**2)) * torch.sin(np.pi * x)
        return trial_dx2 + trial_dy2 - RHS


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            x, y = batch[:,0].view(-1,1), batch[:,1].view(-1,1)
            y1 = torch.ones_like(y)
            # print(y)
            # print(y1)
            xy1 = torch.cat((x,y1),1)
            # print(xy1)
            n_xy = network(batch).view(-1,1)
            # print(n_xy)
            n_x1 = network(xy1).view(-1,1)
            # print(n_x1)

            dn = torch.autograd.grad(n_xy, batch, torch.ones_like(n_xy), retain_graph=True, create_graph=True)[0]
            dn2 = torch.autograd.grad(dn, batch, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
            dndx, dndy = dn[:,0].view(-1,1), dn[:,1].view(-1,1)
            dndx2, dndy2 = dn2[:,0].view(-1,1), dn[:,1].view(-1,1)

            dn_x1 = torch.autograd.grad(n_x1, xy1, torch.ones_like(n_x1), retain_graph=True, create_graph=True)[0]
            dn2_x1 = torch.autograd.grad(dn_x1, xy1, torch.ones_like(dn_x1), retain_graph=True, create_graph=True)[0]
            # print(dn_x1)
            dndx_x1, dndy_x1 = dn_x1[:,0].view(-1,1), dn_x1[:,1].view(-1,1)
            dndx2_x1, dndy2_x1 = dn2_x1[:,0].view(-1,1), dn2_x1[:,1].view(-1,1)
            # print(dndy_x1)

            dndydx_x1 = torch.autograd.grad(dndy_x1, x, torch.ones_like(dndy_x1), retain_graph=True, create_graph=True)[0]
            dndydx2_x1 = torch.autograd.grad(dndydx_x1, x, torch.ones_like(dndydx_x1), retain_graph=True, create_graph=True)[0]

            trial_dx2 = diffEq.dx2_trial(x,y,n_xy, n_x1, dndy_x1, dndx, dndx_x1, dndydx_x1, dndx2, dndx2_x1, dndydx2_x1)
            trial_dy2 = diffEq.dy2_trial(x,y,dndy,dndy2)
            
            D = diffEq.diffEq(x,y,trial_dx2,trial_dy2)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))
            
        
            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        # if epoch%(epochs/5)==0:
        if epoch == epochs:
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange)
        cost_list.append(loss.detach().numpy())

    network.train(False)
    return cost_list


def plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    x_lin  = torch.linspace(xrange[0],xrange[1],num_samples, requires_grad=True)
    y_lin  = torch.linspace(yrange[0],yrange[1],num_samples, requires_grad=True)
    X,Y = torch.meshgrid(x_lin,y_lin)
    x, y  = X.reshape(-1,1), Y.reshape(-1,1)
    xy = torch.cat((x,y),1)
    n_xy = network.forward(xy).view(-1,1)

    y1 = torch.ones_like(y)
    # print(y)
    # print(y1)
    xy1 = torch.cat((x,y1),1)
    # print(xy1)
    # print(n_xy)
    n_x1 = network(xy1).view(-1,1)
    # print(n_x1)

    dn = torch.autograd.grad(n_xy, xy, torch.ones_like(n_xy), retain_graph=True, create_graph=True)[0]
    dn2 = torch.autograd.grad(dn, xy, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
    dndx, dndy = dn[:,0].view(-1,1), dn[:,1].view(-1,1)
    dndx2, dndy2 = dn2[:,0].view(-1,1), dn[:,1].view(-1,1)

    dn_x1 = torch.autograd.grad(n_x1, xy1, torch.ones_like(n_x1), retain_graph=True, create_graph=True)[0]
    dn2_x1 = torch.autograd.grad(dn_x1, xy1, torch.ones_like(dn_x1), retain_graph=True, create_graph=True)[0]
    # print(dn_x1)
    dndx_x1, dndy_x1 = dn_x1[:,0].view(-1,1), dn_x1[:,1].view(-1,1)
    dndx2_x1, dndy2_x1 = dn2_x1[:,0].view(-1,1), dn2_x1[:,1].view(-1,1)
    # print(dndy_x1)

    ddn_x1 = torch.autograd.grad(dn_x1, xy, torch.ones_like(dn_x1), retain_graph = True, create_graph = True)[0]
    d2dn_x1 = torch.autograd.grad(ddn_x1, xy, torch.ones_like(ddn_x1), retain_graph = True, create_graph = True)[0]
    dndydx_x1 = ddn_x1

    # dndydx_x1 = torch.autograd.grad(dndy_x1, x, torch.ones_like(dndy_x1), retain_graph=True, create_graph=True)[0]
    # dndydx2_x1 = torch.autograd.grad(dndydx_x1, x, torch.ones_like(dndydx_x1), retain_graph=True, create_graph=True)[0]

    trial = diffEq.trial(x,y,n_xy,n_x1,dndy_x1)
    exact = diffEq.solution(x,y).detach().numpy()
    trial = trial.detach().numpy()
    surfaceLoss = ((trial-exact)**2).mean()
    print("mean square difference between trial and exact solution = ", surfaceLoss)

    trial = trial.reshape(num_samples,num_samples)

    X = X.detach().numpy()
    Y = Y.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X,Y,trial,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(X,Y,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()

    return surfaceLoss


num_samples = 10
xrange = [0,1]
yrange = [0,1]

epochs = 5000
# lrs = [(7e-3 + i * 1e-4) for i in range(11)]
# lrs = [(4e-3 + i * 5e-4) for i in range(1,11)]
# lrs = [(i * 1e-3) for i in range(1,11)]
lrs = [1e-2]
finalLosses = []
surfaceLosses = []
for lr in lrs:
    losses = [1]
    iterations = 0
    network     = Fitter(num_hidden_nodes=10)
    loss_fn      = torch.nn.MSELoss()
    optimiser  = torch.optim.Adam(network.parameters(), lr = lr)
    train_set    = DataSet(xrange,yrange,num_samples)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=50, shuffle=True)
    diffEq = DiffEq(xrange, yrange, num_samples)
    while losses[-1] > 0.0001  and iterations < 10:
        newLoss = train(network, train_loader, loss_fn,
                            optimiser, diffEq, epochs, iterations)
        losses.extend(newLoss)
        iterations += 1
    losses = losses[1:]
    finalLoss = losses[-1]
    finalLosses.append(finalLoss)
    print("lr = ", lr)
    print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

    surfaceLoss = plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])
    surfaceLosses.append(surfaceLoss)

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()

plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])

# plt.semilogy(lrs,surfaceLosses)
# plt.xlabel("Learning Rate")
# plt.ylabel("Mean Squared Error ")
# plt.title("Mean Squared Error of Network from Exact Solution")
# plt.show()

# plt.semilogy(lrs,finalLosses)
# plt.xlabel("Learning Rate")
# plt.ylabel("Final Loss")
# plt.title("Effect of Learning Rate on Final Loss")
# plt.show()


#%%