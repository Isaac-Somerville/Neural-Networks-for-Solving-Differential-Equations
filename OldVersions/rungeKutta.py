#%%
import torch
import torch.utils.data
import torch.distributions
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

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

class DataSet(torch.utils.data.Dataset):
    """Creates range of evenly-spaced x- and y-coordinates as test data"""
    def __init__(self, xRange, yRange, uRange, vRange, tRange, numSamples):
        X = torch.distributions.uniform.Uniform(xRange[0],xRange[1]).sample([numSamples,1])
        Y = torch.distributions.uniform.Uniform(yRange[0],yRange[1]).sample([numSamples,1])
        U = torch.distributions.uniform.Uniform(uRange[0],uRange[1]).sample([numSamples,1])
        V = torch.distributions.uniform.Uniform(vRange[0],vRange[1]).sample([numSamples,1])
        T = torch.distributions.uniform.Uniform(tRange[0],tRange[1]).sample([numSamples,1])

        # input of forward function must have shape (batch_size, 5)
        self.data_in = torch.cat((X,Y,U,V,T),1)

    def __len__(self):
        return self.data_in.shape[0]

    def __getitem__(self, i):
        return self.data_in[i]

xRange = [1.05,1.052]
yRange = [0.099, 0.101]
uRange = [-0.5,-0.4]
vRange = [-0.3,-0.2]
tRange = [-0.01,5]
numSamples = 20

batch = DataSet(xRange,yRange,uRange,vRange,tRange,numSamples).data_in

xList = batch[:,0].view(-1,1)
yList = batch[:,1].view(-1,1)
uList = batch[:,2].view(-1,1)
vList = batch[:,3].view(-1,1)


t0 = -0.01
tFinal = 5
timeStep = 0.01
mu = 0.01

for i in range(len(xList)):
    x0 = xList[i]
    y0 = yList[i]
    u0 = uList[i]
    v0 = vList[i]
    xs, ys = rungeKutta(x0, y0, u0, v0, t0, mu, tFinal, timeStep)
    plt.plot(xs,ys, color = 'r')
plt.plot([0.],[0.], marker = '.', markersize = 40)
plt.plot([1.],[0.], marker = '.', markersize = 10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#%%