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
        X = torch.linspace(xrange[0], xrange[1], num_samples, requires_grad=True)
        Y = torch.linspace(yrange[0], yrange[1], num_samples, requires_grad=True)
        # create tuple of (num_samples x num_samples) points
        x, y = torch.meshgrid(X, Y)

        # input of forward function must have shape (batch_size, 2)
        # self.data_in = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), 1)

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
        hidden = torch.tanh(self.fc1(input))
        z = self.fc2(hidden)
        return z


class DiffEq:
    """
    Differential equation from Lagaris et al. problem 5
    This problem is a PDE in two variables, with Dirichlet
    boundary conditions
    """

    def __init__(self, xrange, yrange, num_samples):
        self.xrange = xrange
        self.yrange = yrange
        self.num_samples = num_samples

    def solution(self, x, y):
        return y**2 * torch.sin(np.pi * x)

    def trial(self, x, y, n_xy, n_xy1, dy_n_xy1):
        # trial_term guarantees to satisfy boundary conditions
        trial_term = y * 2 * torch.sin(np.pi * x)
        return trial_term + x * (1 - x) * y * (n_xy - n_xy1 - dy_n_xy1)

    def diffEq(self, x, y, trial):
        dx_trial = grad(
            trial, x, torch.ones_like(trial), create_graph=True, retain_graph=True
        )[0]
        dx2_trial = grad(dx_trial, x, torch.ones_like(dx_trial), create_graph=True)[0]
        dy_trial = grad(trial, y, torch.ones_like(trial), create_graph=True)[0]
        # print(dy_trial)
        dy2_trial = grad(
            dy_trial,
            y,
            torch.ones_like(dy_trial),
            create_graph=True,
            retain_graph=True
        )[0]
        # if dy2_trial == None:
        #     dy2_trial = torch.zeros_like(dy_trial, requires_grad=True)
        # print(dy2_trial)
        RHS = (2 - (np.pi * y) ** 2) * torch.sin(np.pi * x)
        return dx2_trial + dy2_trial - RHS


def train(network, loader, loss_fn, optimiser, diffEq, epochs, iterations):
    """Trains the neural network"""
    cost_list = []
    network.train(True)
    for epoch in range(epochs + 1):
        for batch in loader:
            # x, y = batch[:, 0].view(-1, 1), batch[:, 1].view(-1, 1)
            n_xy = network(batch).view(-1, 1)
            x,y = torch.split(batch,1,1)
            print(batch)
            print(x)

            y1 = torch.ones_like(y, requires_grad=True)
            xy1 = torch.cat((x, y1), 1)
            n_xy1 = network(xy1).view(-1, 1)
            # d_n_xy1 = grad(n_xy1, xy1, torch.ones_like(n_xy1), create_graph=True)[0]
            # print(d_n_xy1.shape)
            # print(d_n_xy1)
            # dx_n_xy1, dy_n_xy1 = torch.split(d_n_xy1,1,1)
            # print(d_n_xy1)
            # print(dy_n_xy1)

            dy_n_xy1 = grad(n_xy1, y1, torch.ones_like(n_xy1), create_graph=True)[0]
            print(dy_n_xy1)

            # Get value of trial solution f(x,y)
            trial = diffEq.trial(x, y, n_xy, n_xy1, dy_n_xy1)

            # Get value of diff equations D(x,y) = 0
            D = diffEq.diffEq(x, y, trial)

            # Calculate and store loss
            loss = loss_fn(D, torch.zeros_like(D))

            # Optimization algorithm
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()

        if epoch % (epochs / 5) == 0:
            plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange)

        # store final loss of each epoch
        cost_list.append(loss.detach().numpy())

    network.train(False)
    return cost_list


def plotNetwork(network, diffEq, epoch, epochs, iterations, xrange, yrange):
    """
    Plots the outputs of both neural networks, along with the
    analytic solution in the same range
    """
    x_lin = torch.linspace(xrange[0], xrange[1], num_samples, requires_grad=True)
    y_lin = torch.linspace(yrange[0], yrange[1], num_samples, requires_grad=True)
    X, Y = torch.meshgrid(x_lin, y_lin)
    x, y = X.reshape(-1, 1), Y.reshape(-1, 1)
    xy = torch.cat((x, y), 1)
    n_xy = network.forward(xy)

    y1 = torch.ones_like(y, requires_grad=True)
    xy1 = torch.cat((x, y1), 1)
    n_xy1 = network.forward(xy1)
    # d_n_xy1 = grad(n_xy1, xy1, torch.ones_like(n_xy1), create_graph=True)[0]
    # dy_n_xy1 = d_n_xy1[:, 1].view(-1, 1)
    dy_n_xy1 = grad(n_xy1, y1, torch.ones_like(n_xy1), create_graph=True)[0]

    trial = diffEq.trial(x, y, n_xy, n_xy1, dy_n_xy1)
    exact = diffEq.solution(x, y).detach().numpy()
    trial = trial.detach().numpy()
    print(
        "mean square difference between trial and exact solution = ",
        ((trial - exact) ** 2).mean(),
    )

    trial = trial.reshape(num_samples, num_samples)

    X = X.detach().numpy()
    Y = Y.detach().numpy()
    ax = plt.axes(projection="3d")
    surf = ax.plot_surface(
        X, Y, trial, rstride=1, cstride=1, cmap="plasma", edgecolors="none"
    )
    # surf._facecolors2d = surf._facecolor3d
    # surf._edgecolors2d = surf._edgecolor3d
    # plt.colorbar(surf, location = 'left')
    ax.scatter(X, Y, exact, label="Exact Solution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.set_title(str(epoch + iterations * epochs) + " Epochs")
    plt.show()


network = Fitter(num_hidden_nodes=10)
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.Adam(network.parameters(), lr=1e-2)


ranges = [[0, 1]]
for xrange in ranges:
    yrange = xrange
    num_samples = 10
    diffEq = DiffEq(xrange, yrange, num_samples)
    train_set = DataSet(xrange, yrange, num_samples)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=32, shuffle=True
    )

    losses = [1]
    iterations = 0
    epochs = 5000
    while losses[-1] > 0.001 and iterations < 10:
        newLoss = train(
            network, train_loader, loss_fn, optimiser, diffEq, epochs, iterations
        )
        losses.extend(newLoss)
        iterations += 1
    losses = losses[1:]
    print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

    plt.semilogy(losses)
    plt.xlabel("Epochs")
    plt.ylabel("Log of Loss")
    plt.title("Loss")
    plt.show()

plotNetwork(network, diffEq, 0, epochs, iterations, [0, 1], [0, 1])

#%%
