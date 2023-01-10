#%%
import torch
import torch.utils.data
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, xRange, yRange, uRange, vRange, tRange, numSamples):
        X = torch.distributions.uniform.Uniform(xRange[0],xRange[1]).sample([numSamples,1])
        Y = torch.distributions.uniform.Uniform(yRange[0],yRange[1]).sample([numSamples,1])
        U = torch.distributions.uniform.Uniform(uRange[0],uRange[1]).sample([numSamples,1])
        V = torch.distributions.uniform.Uniform(vRange[0],vRange[1]).sample([numSamples,1])
        T = torch.distributions.uniform.Uniform(tRange[0],tRange[1]).sample([numSamples,1])
        X.requires_grad_()
        Y.requires_grad_()
        U.requires_grad_()
        V.requires_grad_()
        T.requires_grad_()

        # input of forward function must have shape (batch_size, 5)
        self.data_in = torch.cat((X,Y,U,V,T),1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, i):
        return self.data_in[i]

class Fitter(torch.nn.Module):
    """Forward propagations"""
    def __init__(self, numHiddenNodes,numHiddenLayers):
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(5, numHiddenNodes)
        self.fcs = [torch.nn.Linear(numHiddenNodes,numHiddenNodes) for _ in range(numHiddenLayers)]
        self.fcLast = torch.nn.Linear(numHiddenNodes, 4)

    def forward(self, input):
        hidden = torch.tanh(self.fc1(input))
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        out = self.fcLast(hidden)
        out = out.transpose(0,1)
        xOut, yOut, uOut, vOut = out[0].view(-1,1), out[1].view(-1,1), out[2].view(-1,1), out[3].view(-1,1)
        return xOut, yOut, uOut, vOut

class DiffEq():
    """
    Differential equations from Flamant et al. for Planar Three Body Problem
    """
    def __init__(self,x,y,u,v,mu):
        self.x = x
        self.y = y
        self.u = u
        self.v = v
        self.mu = mu

    def xTrial(self,xOut,t):
        return self.x + (1 - torch.exp(-t)) * xOut

    def yTrial(self,yOut,t):
        return self.y + (1 - torch.exp(-t)) * yOut

    def uTrial(self,uOut,t):
        return self.u + (1 - torch.exp(-t)) * uOut

    def vTrial(self,vOut,t):
        return self.v + (1 - torch.exp(-t)) * vOut

    def dxEq(self, uOut, xOut,t):
        u = self.uTrial(uOut,t)
        xTrial = self.xTrial(xOut,t)
        dxTrial = grad(xTrial,t,torch.ones_like(t), create_graph=True)[0]
        return abs(dxTrial - u)**2

    def dyEq(self, vOut, yOut,t):
        v = self.vTrial(vOut,t)
        yTrial = self.yTrial(yOut,t)
        dyTrial = grad(yTrial,t,torch.ones_like(t), create_graph=True)[0]
        return abs(dyTrial - v)**2
    
    def duEq(self,xOut,yOut,vOut,uOut,t):
        x = self.xTrial(xOut,t)
        y = self.yTrial(yOut,t)
        v = self.vTrial(vOut,t)
        uTrial = self.uTrial(uOut,t)
        duTrial = grad(uTrial,t,torch.ones_like(t), create_graph=True)[0]
        return abs(duTrial - (x - self.mu + 2*v - 
                    (((self.mu*(x-1)) / ((x-1)**2 + y**2)**(3/2)) 
                        + ((1-self.mu)*x / (x**2 + y**2)**(3/2)))
                         )
                )**2

    def dvEq(self,xOut,yOut,uOut,vOut,t):
        x = self.xTrial(xOut,t)
        y = self.yTrial(yOut,t)
        u = self.uTrial(uOut,t)
        vTrial = self.vTrial(vOut,t)
        dvTrial = grad(vTrial,t,torch.ones_like(t), create_graph=True)[0]
        return abs(dvTrial - (y - 2*u - 
                    (((self.mu * y) / ((x-1)**2 + y**2)**(3/2))
                        +((1-self.mu)*y / (x**2 + y**2)**(3/2) ))
                        )
                )**2

    def totalDiffEq(self,xOut,yOut,uOut,vOut,t):
        return (self.dxEq(uOut,xOut,t) + self.dyEq(vOut,yOut,t) 
                + self.duEq(xOut,yOut,vOut,uOut,t)
                + self.dvEq(xOut,yOut,uOut,vOut,t))

def train(network, lossFn, optimiser, xRange,yRange,uRange,vRange,tRange,
            numSamples, mu, lmbda, epochs, iterations):
    """Trains the neural network"""
    costList=[]
    network.train(True)
    for epoch in range(epochs+1):
        batch = DataSet(xRange,yRange,uRange,vRange,tRange,numSamples).data_in
        x = batch[0,:].view(-1,1)
        y = batch[1,:].view(-1,1)
        u = batch[2,:].view(-1,1)
        v = batch[3,:].view(-1,1)
        t = batch[4,:].view(-1,1)

        xOut, yOut, uOut, vOut = network(batch)
        # print(nOut)

        # xOut = nOut[0,:].view(-1,1)
        # yOut = nOut[1,:].view(-1,1)
        # uOut = nOut[2,:].view(-1,1)
        # vOut = nOut[3,:].view(-1,1)

        diffEq = DiffEq(x, y, u, v, mu)

        # backpropagation
        # dxOut = grad(xOut,t,torch.ones_like(t), create_graph=True)[0]
        # dyOut = grad(yOut,t,torch.ones_like(t), create_graph=True)[0]
        # duOut = grad(uOut,t,torch.ones_like(t), create_graph=True)[0]
        # dvOut = grad(vOut,t,torch.ones_like(t), create_graph=True)[0]

        # calculate loss
        D = torch.exp(-lmbda*t) * diffEq.totalDiffEq(xOut,yOut,uOut,vOut,t) / numSamples
        loss = lossFn(D, torch.zeros_like(D))

        # optimisation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if epoch == epochs:
            plotNetwork(network, mu, epoch, epochs, iterations, xRange, yRange, uRange,vRange,tRange)

        #store final loss of each epoch
        costList.append(loss.detach().numpy())

    network.train(False)
    return costList


def plotNetwork(network, mu, epoch, epochs, iterations, 
                xRange, yRange,uRange,vRange,tRange):
    batch = DataSet(xRange,yRange,uRange,vRange,tRange,10).data_in
    t = torch.linspace(tRange[0],tRange[1],10,requires_grad=True).view(-1,1)
    for i in range(len(batch)):
        x = torch.tensor([batch[i][0] for _ in range(10)]).view(-1,1)
        y = torch.tensor([batch[i][1] for _ in range(10)]).view(-1,1)
        u = torch.tensor([batch[i][2] for _ in range(10)]).view(-1,1)
        v = torch.tensor([batch[i][3] for _ in range(10)]).view(-1,1)
        diffEq = DiffEq(x, y, u, v, mu)
        input = torch.cat((x,y,u,v,t),dim=1)
        # print(input)
        xOut, yOut, uOut, vOut = network(input)
        xTrial = diffEq.xTrial(xOut,t).detach().numpy()
        yTrial = diffEq.yTrial(yOut,t).detach().numpy()
        # print(xTrial)
        # print(yTrial)
        plt.plot(xTrial,yTrial)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()





    


xRange = [1.05,1.052]
yRange = [0.099, 0.101]
uRange = [-0.5,-0.4]
vRange = [-0.3,-0.2]
tRange = [-0.01,5]
numSamples = 5
mu = 0.01
lmbda = 2

network    = Fitter(numHiddenNodes=8, numHiddenLayers=4)
lossFn    = torch.nn.MSELoss()
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-2)

losses = [1]
iterations = 0
epochs = 5000
while losses[-1] > 0.001  and iterations < 10:
    newLoss = train(network, lossFn, optimiser, xRange,yRange,uRange,vRange,tRange,
            numSamples, mu, lmbda, epochs, iterations)
    losses.extend(newLoss)
    iterations += 1
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")
plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")
plt.show()

#%%