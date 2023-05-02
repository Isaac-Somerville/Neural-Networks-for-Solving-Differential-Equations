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

    def trial_term(self,x,y):
        """
        First term in trial solution that helps to satisfy BCs
        """
        e_inv = np.exp(-1)
        return ((1-x)*(y**3) + x*(1+(y**3))*e_inv + (1-y)*x*(torch.exp(-x)-e_inv) +
                y * ((1+x)*torch.exp(-x) - (1-x+(2*x*e_inv))))

    def dx_trial(self,x,y,n_out,dndx):
        return (-torch.exp(-x) * (x+y-1) - y**3 + (y**2 +3) * y * np.exp(-1) + y
                + y*(1-y)* ((1-2*x) * n_out + x*(1-x)*dndx))

    def dx2_trial(self,x,y,n_out,dndx,dndx2):
        return (torch.exp(-x) * (x+y-2)
                + y*(1-y)* ((-2*n_out) + 2*(1-2*x)*dndx) + x*(1-x)*dndx2)

    def dy_trial(self,x,y,n_out, dndy):
        return (3*x*(y**2 +1) *np.exp(-1) - (x-1)*(3*(y**2)-1) + torch.exp(-x)
                + x*(1-x)* ((1-2*y) * n_out + y*(1-y)*dndy) )

    def dy2_trial(self,x,y,n_out,dndy,dndy2):
        return (np.exp(-1) * 6 * y * (-np.exp(1)*x + x + np.exp(1))
                + x*(1-x)* ((-2*n_out) + 2*(1-2*y)*dndy) + y*(1-y)*dndy2)
    
    def trial(self,x,y,n_out):
        return self.trial_term(x,y) + x*(1-x)*y*(1-y)*n_out
    
    def diffEq(self,x,y,trial_dx2,trial_dy2):
        RHS = torch.exp(-x) * (x - 2 + y**3 + 6*y)
        return trial_dx2 + trial_dy2 - RHS


def train(network, loader, loss_fn, optimiser, scheduler, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list=[]
    network.train(True)
    for epoch in range(epochs+1):
        for batch in loader:
            n_out = network(batch)

            dn = torch.autograd.grad(n_out, batch, torch.ones_like(n_out), retain_graph=True, create_graph=True)[0]
            dn2 = torch.autograd.grad(dn, batch, torch.ones_like(dn), retain_graph=True, create_graph=True)[0]
            
            # Get first derivatives of NN output: dn/dx , dn/dy
            dndx, dndy = dn[:,0].view(-1,1), dn[:,1].view(-1,1)

            # Get second derivatives of NN output: d^2 n / dx^2,  d^2 n / dy^2
            dndx2, dndy2 = dn2[:,0].view(-1,1), dn2[:,1].view(-1,1)

            x, y = batch[:,0].view(-1,1), batch[:,1].view(-1,1)
            # Get value of trial solution f_{xx}(x,y) and f_{yy}(x,y)
            trial_dx2 = diffEq.dx2_trial(x,y,n_out,dndx,dndx2)
            trial_dy2 = diffEq.dy2_trial(x,y,n_out,dndy,dndy2)
    
            # Get value of diff equations D(x) = 0
            D = diffEq.diffEq(x,y, trial_dx2, trial_dy2)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))
        
            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            scheduler.step(loss)
            
        # if epoch%(epochs/5)==0:
        if epoch == epochs:
            print("loss = ", loss.detach().numpy())
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange)
        
        cost_list.append(loss.detach().numpy())

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

    # Format input into correct shape
    input = torch.cat((x.reshape(-1,1),y.reshape(-1,1)),1)

    # Get output of neural network
    N = network.forward(input)

    # Get trial solution, put into correct shape
    output = diffEq.trial(x.reshape(-1,1),y.reshape(-1,1),N)
    output = output.reshape(num_samples,num_samples).detach().numpy()

    # Get exact solution
    exact = diffEq.solution(x,y).detach().numpy()

    # Calculate residual error
    surfaceLoss = ((output-exact)**2).mean()
    print("mean square difference between trial and exact solution = ", surfaceLoss) 

    # Plot both trial and exact solutions
    x = x.detach().numpy()
    y = y.detach().numpy()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x,y,output,rstride=1, cstride=1,
                cmap='plasma', edgecolor='none')
    ax.scatter(x,y,exact, label = 'Exact Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    ax.set_title(str(epoch + iterations*epochs) + " Epochs")
    #ax.view_init(30, 315)
    plt.show()
    return surfaceLoss



# lrs = [(1e-2 + i * 4e-4) for i in range(10)]
# lrs = [(5e-3 + i * 1e-3) for i in range(1,11)]
# lrs = [(i * 1e-3) for i in range(1,11)]
lrs = [1e-3, 5e-3, 1e-2]
finalLosses = []
surfaceLosses = []
xrange = [0,1]
yrange = [0,1]
for lr in lrs:
    num_samples = 10
    diffEq = DiffEq(xrange, yrange, num_samples)
    network     = Fitter(num_hidden_nodes=10)
    loss_fn      = torch.nn.MSELoss()
    optimiser  = torch.optim.Adam(network.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor=0.1, patience=500, threshold=1e-4, cooldown=0, min_lr=0, eps=1e-10, verbose=True)
    train_set    = DataSet(xrange,yrange,num_samples)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=50, shuffle=True)

    losses = [1]
    iterations = 0
    epochs = 5000
    while losses[-1] > 1e-3 and iterations < 5:
        newLoss = train(network, train_loader, loss_fn,
                            optimiser, scheduler, diffEq, epochs, iterations)
        losses.extend(newLoss)
        iterations += 1
    losses = losses[1:]
    finalLoss = losses[-1]
    finalLosses.append(finalLoss)
    print("lr = ", lr)
    print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()

    surfaceLoss = plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])
    surfaceLosses.append(surfaceLoss)


plotNetwork(network, diffEq, 0, epochs, iterations, [0,1], [0,1])
plt.semilogy(lrs,surfaceLosses)
plt.xlabel("Learning Rate")
plt.ylabel("Mean Squared Error ")
plt.title("Mean Squared Error of Network from Exact Solution")
plt.show()

plt.semilogy(lrs,finalLosses)
plt.xlabel("Learning Rate")
plt.ylabel("Final Loss")
plt.title("Effect of Learning Rate on Final Loss")
plt.show()

#%%