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
        # print(out)
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
        return dxTrial - u

    def dyEq(self, vOut, yOut,t):
        v = self.vTrial(vOut,t)
        yTrial = self.yTrial(yOut,t)
        dyTrial = grad(yTrial,t,torch.ones_like(t), create_graph=True)[0]
        return dyTrial - v
    
    def duEq(self,xOut,yOut,vOut,uOut,t):
        x = self.xTrial(xOut,t)
        y = self.yTrial(yOut,t)
        v = self.vTrial(vOut,t)
        uTrial = self.uTrial(uOut,t)
        duTrial = grad(uTrial,t,torch.ones_like(t), create_graph=True)[0]
        return duTrial - (x - self.mu + 2*v - 
                    (((self.mu*(x-1)) / ((x-1)**2 + y**2)**(3/2)) 
                        + ((1-self.mu)*x / (x**2 + y**2)**(3/2)))
                         )
                

    def dvEq(self,xOut,yOut,uOut,vOut,t):
        x = self.xTrial(xOut,t)
        y = self.yTrial(yOut,t)
        u = self.uTrial(uOut,t)
        vTrial = self.vTrial(vOut,t)
        dvTrial = grad(vTrial,t,torch.ones_like(t), create_graph=True)[0]
        return dvTrial - (y - 2*u - 
                    (((self.mu * y) / ((x-1)**2 + y**2)**(3/2))
                        +((1-self.mu)*y / (x**2 + y**2)**(3/2) ))
                        )
                

    # def totalDiffEq(self,xOut,yOut,uOut,vOut,t):
    #     return (self.dxEq(uOut,xOut,t) + self.dyEq(vOut,yOut,t) 
    #             + self.duEq(xOut,yOut,vOut,uOut,t)
    #             + self.dvEq(xOut,yOut,uOut,vOut,t))

def train(network, lossFn, optimiser, xRange,yRange,uRange,vRange,tRange,
            numSamples, mu, lmbda, epochs, iterations, numTimeSteps):
    """Trains the neural network"""
    costList=[]
    network.train(True)
    for epoch in range(epochs+1):
        batch = DataSet(xRange,yRange,uRange,vRange,tRange,numSamples).data_in
        x = batch[:,0].view(-1,1)
        y = batch[:,1].view(-1,1)
        u = batch[:,2].view(-1,1)
        v = batch[:,3].view(-1,1)
        t = batch[:,4].view(-1,1)

        xOut, yOut, uOut, vOut = network(batch)

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
        dxEq = diffEq.dxEq(uOut, xOut,t)
        dyEq = diffEq.dyEq(vOut, yOut,t)
        duEq = diffEq.duEq(xOut,yOut,vOut,uOut,t)
        dvEq = diffEq.dvEq(xOut,yOut,uOut,vOut,t)
        # D = torch.exp(-lmbda*t) * diffEq.totalDiffEq(xOut,yOut,uOut,vOut,t)
        dxLoss = lossFn( torch.exp(-lmbda*t) * dxEq, torch.zeros_like(dxEq))
        dyLoss = lossFn( torch.exp(-lmbda*t) * dyEq, torch.zeros_like(dyEq))
        duLoss = lossFn( torch.exp(-lmbda*t) * duEq, torch.zeros_like(duEq))
        dvLoss = lossFn( torch.exp(-lmbda*t) * dvEq, torch.zeros_like(dvEq))
        loss = (dxLoss + dyLoss + duLoss + dvLoss)

        # optimisation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if epoch == epochs:
            plotNetwork(network, mu, epoch, epochs, iterations, 
                        xRange, yRange, uRange,vRange,tRange, numTimeSteps)

        #store final loss of each epoch
        costList.append(loss.detach().numpy())

    network.train(False)
    return costList

def plotNetwork(network, mu, epoch, epochs, iterations, 
                xRange, yRange,uRange,vRange,tRange, numTimeSteps):
    timeStep = (tRange[1] - tRange[0]) / numTimeSteps
    batch = DataSet(xRange,yRange,uRange,vRange,tRange,10).data_in
    t = torch.linspace(tRange[0],tRange[1],numTimeSteps,requires_grad=True).view(-1,1)

    for i in range(len(batch)):
        x = torch.tensor([batch[i][0] for _ in range(numTimeSteps)]).view(-1,1)
        y = torch.tensor([batch[i][1] for _ in range(numTimeSteps)]).view(-1,1)
        u = torch.tensor([batch[i][2] for _ in range(numTimeSteps)]).view(-1,1)
        v = torch.tensor([batch[i][3] for _ in range(numTimeSteps)]).view(-1,1)
        diffEq = DiffEq(x, y, u, v, mu)
        input = torch.cat((x,y,u,v,t),dim=1)
        # for j in range(5):
        #     print(input[j])
        # print(input)
        xOut, yOut, uOut, vOut = network(input)
        xTrial = diffEq.xTrial(xOut,t).detach().numpy()
        # print(len(xTrial))
        yTrial = diffEq.yTrial(yOut,t).detach().numpy()
        # print(xTrial)
        # print(yTrial)
        plt.plot(xTrial,yTrial, color = 'b')

        # Plot Runge-Kutta solution
        x0 = batch[i][0].item()
        y0 = batch[i][1].item()
        u0 = batch[i][2].item()
        v0 = batch[i][3].item()
        xExact, yExact = rungeKutta(x0, y0, u0, v0, tRange[0], mu, tRange[1], timeStep)

        plt.plot(xExact, yExact, color = 'r')

    plt.plot([0.],[0.], marker = '.', markersize = 40)
    plt.plot([1.],[0.], marker = '.', markersize = 10)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.xlim((-1,1.1))
    # plt.ylim((-1,1))
    plt.title(str(epoch + iterations*epochs) + " Epochs")
    plt.show()

def rungeKutta(x0, y0, u0, v0, t0, mu, tFinal, timeStep):
    """
    Implements 4th-Order Runge-Kutta Method to evaluate
    system of ODEs from time t0 to time tFinal
    """
    # System of ODEs
    def dxdt(x,y,u,v,mu):
        return u

    def dydt(x,y,u,v,mu):
        return v

    def dudt(x,y,u,v,mu):
        return (x - mu + 2*v - 
                (((mu*(x-1)) / ((x-1)**2 + y**2)**(3/2)) 
                    + ((1-mu)*x / (x**2 + y**2)**(3/2)))
                        )

    def dvdt(x,y,u,v,mu):
        return (y - 2*u - 
                (((mu * y) / ((x-1)**2 + y**2)**(3/2))
                    +((1-mu)*y / (x**2 + y**2)**(3/2) ))
                    )

    # Count number of iterations using length of timeStep
    n = (int)((tFinal - t0)/timeStep)

    # Iterate for number of iterations
    x = x0
    y = y0
    u = u0
    v = v0
    xList = [x0]
    yList = [y0]
    for _ in range(n):
        "Apply Runge Kutta Formulas to find next value of x, y, u, v"
        a1 = timeStep * dxdt(x,y,u,v,mu)
        b1 = timeStep * dydt(x,y,u,v,mu)
        c1 = timeStep * dudt(x,y,u,v,mu)
        d1 = timeStep * dvdt(x,y,u,v,mu)
        
        a2 = timeStep * dxdt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)
        b2 = timeStep * dydt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)
        c2 = timeStep * dudt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)        
        d2 = timeStep * dvdt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)

        a3 = timeStep * dxdt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)
        b3 = timeStep * dydt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)
        c3 = timeStep * dudt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)        
        d3 = timeStep * dvdt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)

        a4 = timeStep * dxdt(x + a3, y + b3, u + c3, v + d3, mu)
        b4 = timeStep * dydt(x + a3, y + b3, u + c3, v + d3, mu)
        c4 = timeStep * dudt(x + a3, y + b3, u + c3, v + d3, mu)        
        d4 = timeStep * dvdt(x + a3, y + b3, u + c3, v + d3, mu)

        # Update next value of x, y, u, v
        x = x + (1.0 / 6.0)*(a1 + 2 * a2 + 2 * a3 + a4)
        y = y + (1.0 / 6.0)*(b1 + 2 * b2 + 2 * b3 + b4)
        u = u + (1.0 / 6.0)*(c1 + 2 * c2 + 2 * c3 + c4)
        v = v + (1.0 / 6.0)*(d1 + 2 * d2 + 2 * d3 + d4)

        xList.append(x)
        yList.append(y)
 
        # Update next value of t
        t0 = t0 + timeStep

    return xList, yList


xRange = [1.05,1.052]
yRange = [0.099, 0.101]
uRange = [-0.5,-0.4]
vRange = [-0.3,-0.2]
tRange = [-0.01,5]
numSamples = 5
mu = 0.01
lmbda = 2
numTimeSteps = 1000

network    = Fitter(numHiddenNodes=8, numHiddenLayers=2)
lossFn    = torch.nn.MSELoss()
optimiser  = torch.optim.Adam(network.parameters(), lr = 1e-3)

losses = [1]
iterations = 0
epochs = 5000
while iterations < 20:
    newLoss = train(network, lossFn, optimiser, xRange,yRange,uRange,vRange,tRange,
            numSamples, mu, lmbda, epochs, iterations, numTimeSteps)
    losses.extend(newLoss)
    iterations += 1
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")
plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("Loss")
plt.show()

#%%