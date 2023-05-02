#%%
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import grad

class DataSet(torch.utils.data.Dataset):
    """Samples 'batchSize' random samples of initial values (x_0, y_0, u_0, v_0) and times t from 
        X_0 x tRange where X_0 = xRange x yRange x uRange x vRange, the ranges of initial conditions """
    def __init__(self, xRange, yRange, uRange, vRange, tRange, batchSize):
        """
        Arguments:
        xRange (list of length 2) -- lower and upper limits for initial conditions x_0
        yRange (list of length 2) -- lower and upper limits for initial conditions y_0
        uRange (list of length 2) -- lower and upper limits for initial conditions u_0
        vRange (list of length 2) -- lower and upper limits for initial conditions v_0
        tRange (list of length 2) -- lower and upper limits for time values t
        batchSize (int) -- number of training data samples

        Returns:
        DataSet object with one attribute:
        dataIn (tuple of 5 PyTorch tensor of shape (batchSize,1)) -- 'batchSize' neural network inputs
            of the form (x_0, y_0, u_0, v_0, t)
        """
        global device
        # uniformly sample values of x_0, y_0, u_0, v_0 and t from their respective ranges
        # move all tensors to GPU
        X = torch.distributions.uniform.Uniform(xRange[0],xRange[1]).sample([batchSize,1]).to(device)
        Y = torch.distributions.uniform.Uniform(yRange[0],yRange[1]).sample([batchSize,1]).to(device)
        U = torch.distributions.uniform.Uniform(uRange[0],uRange[1]).sample([batchSize,1]).to(device)
        V = torch.distributions.uniform.Uniform(vRange[0],vRange[1]).sample([batchSize,1]).to(device)
        T = torch.distributions.uniform.Uniform(tRange[0],tRange[1]).sample([batchSize,1]).to(device)
        # set requires grad = True to create a computation graph and allow gradient calculation
        X.requires_grad_()
        Y.requires_grad_()
        U.requires_grad_()
        V.requires_grad_()
        T.requires_grad_()

        # return inputs as a tuple of 5 tensors (these must be concatenated before passed to network)
        self.data_in = (X,Y,U,V,T)

    def __len__(self):
        """
        Arguments:
        None
        
        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return self.data_in.shape[0]

    def __getitem__(self, idx):
        """
        Used by DataLoader object to retrieve training data points
        Arguments:
        idx (int) -- index of data point required

        Returns:
        (x_0, y_0, u_0, v_0, t) (tensor of shape (1,5)) -- data point at index 'idx'
        """
        return self.data_in[idx]

class SolutionBundle(torch.nn.Module):
    """
    A deep neural network object, with 5 nodes in the input layer, 1 node in the 
    output layer, and 'numHiddenLayers' hidden layers each with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes, numHiddenLayers):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layers
        numHiddenLayers (int) -- number of hidden layers

        Returns:
        SolutionBundle object (neural network) with three attributes:
        fc1 (fully connected layer) -- linear transformation of first layer
        fcs (list of fully connected layers) -- linear transformations of hidden layers
        fcLast (fully connected layer) -- linear transformation of outer layer
        """
        super(SolutionBundle, self).__init__()
        # create first layer, apply Xavier initialisation
        self.fc1 = torch.nn.Linear(5, numHiddenNodes)
        self.fc1.apply(self.initWeightsXavier)
        # create list of hidden layers, apply Xavier initialisation
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(numHiddenNodes, numHiddenNodes)
                    for _ in range(numHiddenLayers)])
        self.fcs.apply(self.initWeightsXavier)
        # create final layer, apply Xavier initialisation
        self.fcLast = torch.nn.Linear(numHiddenNodes, 4)
        self.fcLast.apply(self.initWeightsXavier)

    def forward(self, input):
        """
        Function which performs forward propagation in the neural network.

        Arguments:
        input (PyTorch tensor shape (batchSize, 5)) -- input of neural network
        Returns:
        output (PyTorch tensor shape (batchSize, 4)) -- output of neural network
        """
        hidden = torch.tanh(self.fc1(input))
        # pass through all hidden layers
        for i in range(len(self.fcs)):
            hidden = torch.tanh(self.fcs[i](hidden))
        output = self.fcLast(hidden)
        return output
    
    def initWeightsXavier(self, layer):
        """
        Function which initialises weights according to Xavier initialisation
        
        Arguments:
        layer (Linear object) -- weights and biases of a layer
        Returns:
        None
        """
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_uniform_(layer.weight, gain = torch.nn.init.calculate_gain('tanh'))

def trialSolution(varInitial, varOut, t):
    """
    Trial solution for a given variable varOut at times t with initial values varInitial
    
    Arguments:
    varInitial (tensor of shape (batchSize,1)) -- initial values of given variable
    varOut (tensor of shape (batchSize,1)) -- network output for variable at times t
    t (tensor of shape (batchSize,1)) -- times at which variable is evaluated
    
    Returns:
    trialSoln (tensor of shape (batchSize,1)) -- trial solution for given variable at 
        times t with initial values varInitial"""
    trialSoln = varInitial + (1 - torch.exp(-t)) * varOut
    return trialSoln

def dTrialSolution(varOut, dVarOut, t):
    """
    Derivative w.r.t. t of trial solution for a given variable varOut at times t with initial values varInitial
    
    Arguments:
    varOut (tensor of shape (batchSize,1)) -- network output for variable at times t
    dVarOut (tensor of shape (batchSize,1)) -- derivative of network output for variable w.r.t. t at times t
    t (tensor of shape (batchSize,1)) -- times at which variable is evaluated
    
    Returns:
    trialSoln (tensor of shape (batchSize,1)) -- trial solution for given variable at 
        times t with initial values varInitial"""
    dTrialSoln = ((1 - torch.exp(-t)) * dVarOut) + (torch.exp(-t)) * varOut
    return dTrialSoln

def diffEqX(u0, uOut, xOut, dxOut, t):
    """
    Returns LHS of differential equation for x(t)

    Arguments:
    u0 (tensor of shape (batchSize,1)) -- initial values of u at time t_0
    uOut (tensor of shape (batchSize,1)) -- network output for u at times t
    xOut (tensor of shape (batchSize,1)) -- network output for x at times t
    dxOut (tensor of shape (batchSize,1)) -- derivative of network output for x w.r.t. t at times t
    t (tensor of shape (batchSize,1)) -- times

    Returns:
    dxTrial - u (tensor of shape (batchSize,1)) -- LHS of DE for x(t) at times t
    """
    u = trialSolution(u0, uOut, t)
    dxTrial = dTrialSolution(xOut, dxOut, t)
    return dxTrial - u

def diffEqY(v0, vOut, yOut, dyOut, t):
    """
    Returns LHS of differential equation for y(t)

    Arguments:
    v0 (tensor of shape (batchSize,1)) -- initial values of v at time t_0
    vOut (tensor of shape (batchSize,1)) -- network output for v at times t
    yOut (tensor of shape (batchSize,1)) -- network output for y at times t
    dyOut (tensor of shape (batchSize,1)) -- derivative of network output for y w.r.t. t at times t
    t (tensor of shape (batchSize,1)) -- times

    Returns:
    dyTrial - v (tensor of shape (batchSize,1)) -- LHS of DE for y(t) at times t
    """
    v = trialSolution(v0, vOut, t)
    dyTrial = dTrialSolution(yOut, dyOut, t)
    return dyTrial - v

def diffEqU(x0, y0, v0, xOut, yOut, vOut, uOut, duOut, t, mu):
    """
    Returns LHS of differential equation for u(t)

    Arguments:
    x0 (tensor of shape (batchSize,1)) -- initial values of x at time t_0
    y0 (tensor of shape (batchSize,1)) -- initial values of y at time t_0
    v0 (tensor of shape (batchSize,1)) -- initial values of v at time t_0
    xOut (tensor of shape (batchSize,1)) -- network output for x at times t
    yOut (tensor of shape (batchSize,1)) -- network output for y at times t
    uOut (tensor of shape (batchSize,1)) -- network output for u at times t
    vOut (tensor of shape (batchSize,1)) -- network output for v at times t
    duOut (tensor of shape (batchSize,1)) -- derivative of network output for u w.r.t. t at times t
    t (tensor of shape (batchSize,1)) -- times
    mu (float) -- non-dimensionalised mass of the second body

    Returns:
    diffEqULHS (tensor of shape (batchSize,1)) -- LHS of DE for u(t) at times t
    """
    x = trialSolution(x0, xOut, t)
    y = trialSolution(y0, yOut, t)
    v = trialSolution(v0, vOut, t)
    duTrial = dTrialSolution(uOut, duOut, t)
    diffEqULHS = duTrial - (x - mu + 2 * v - (((mu * (x - 1)) / ((x - 1) ** 2 + y**2) ** (3 / 2))
            + (((1 - mu) * x) / (x**2 + y**2) ** (3 / 2))))
    return diffEqULHS

def diffEqV(x0, y0, u0, xOut, yOut, uOut, vOut, dvOut, t, mu):
    """
    Returns LHS of differential equation for v(t)

    Arguments:
    x0 (tensor of shape (batchSize,1)) -- initial values of x at time t_0
    y0 (tensor of shape (batchSize,1)) -- initial values of y at time t_0
    u0 (tensor of shape (batchSize,1)) -- initial values of u at time t_0
    xOut (tensor of shape (batchSize,1)) -- network output for x at times t
    yOut (tensor of shape (batchSize,1)) -- network output for y at times t
    uOut (tensor of shape (batchSize,1)) -- network output for u at times t
    vOut (tensor of shape (batchSize,1)) -- network output for v at times t
    dvOut (tensor of shape (batchSize,1)) -- derivative of network output for v w.r.t. t at times t
    t (tensor of shape (batchSize,1)) -- times
    mu (float) -- non-dimensionalised mass of the second body

    Returns:
    diffEqVLHS (tensor of shape (batchSize,1)) -- LHS of DE for v(t) at times t
    """
    x = trialSolution(x0, xOut, t)
    y = trialSolution(y0, yOut, t)
    u = trialSolution(u0, uOut, t)
    dvTrial = dTrialSolution(vOut, dvOut, t)
    diffEqVLHS = dvTrial - (y - 2 * u - (((mu * y) / ((x - 1) ** 2 + y**2) ** (3 / 2))
            + (((1 - mu) * y) / (x**2 + y**2) ** (3 / 2))))
    return diffEqVLHS

def train(network, lossFn, optimiser, scheduler, xRange, yRange, uRange, vRange, tRange, batchSize, mu, lmbda):
    """
    A function to train a neural network on a batch of size 'batchSize' to approximate the solution to the 
    planar-restricted three-body problem for a bundle of initial conditions

    Arguments:
    network (Module) -- the neural network
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    scheduler (Learning Rate Scheduler) -- reduces learning rate if cost value is plateauing
    xRange (list of length 2) -- lower and upper limits for initial conditions x_0
    yRange (list of length 2) -- lower and upper limits for initial conditions y_0
    uRange (list of length 2) -- lower and upper limits for initial conditions u_0
    vRange (list of length 2) -- lower and upper limits for initial conditions v_0
    tRange (list of length 2) -- lower and upper limits for time values t
    batchSize (int) -- number of training samples to use
    mu (float) -- non-dimensionalised mass of the second body
    lmbda (float) -- factor in the weighting function exp(-lmbda * t) in the cost function

    Returns:
    cost (float) -- network's cost evaluated on single batch of training data"""
    global device
    network.train(True) # set network into training mode
    x, y, u, v, t  = DataSet(xRange,yRange,uRange,vRange,tRange,batchSize).data_in # generate data set
    batch = torch.cat((x,y,u,v,t),1) # input of neural network must be of shape (batchSize, 5)
    out = network.forward(batch) # pass training batch through network
    xOut, yOut, uOut, vOut = torch.split(out, 1, dim = 1) # separate outputs

    # Get derivative of every variable w.r.t. t
    dxOut = grad(xOut,t,torch.ones_like(xOut),retain_graph=True, create_graph=True)[0]
    dyOut = grad(yOut,t,torch.ones_like(yOut),retain_graph=True, create_graph=True)[0]
    duOut = grad(uOut,t,torch.ones_like(uOut),retain_graph=True, create_graph=True)[0]
    dvOut = grad(vOut,t,torch.ones_like(vOut),retain_graph=True, create_graph=True)[0]

    # evaluate each of the 4 differential equations
    dxEq = diffEqX(u, uOut, xOut, dxOut, t)
    dyEq = diffEqY(v, vOut, yOut, dyOut, t)
    duEq = diffEqU(x, y, v, xOut, yOut, vOut, uOut, duOut, t, mu)
    dvEq = diffEqV(x, y, u, xOut, yOut, uOut, vOut, dvOut, t, mu)

    # evaluate cost function with weighting factor exp(-lambda * t)
    dxCost = lossFn( torch.exp(-lmbda*t) * dxEq, torch.zeros_like(dxEq))
    dyCost = lossFn( torch.exp(-lmbda*t) * dyEq, torch.zeros_like(dyEq))
    duCost = lossFn( torch.exp(-lmbda*t) * duEq, torch.zeros_like(duEq))
    dvCost = lossFn( torch.exp(-lmbda*t) * dvEq, torch.zeros_like(dvEq))
    cost = (dxCost + dyCost + duCost + dvCost)

    cost.backward() # perform back propagation
    optimiser.step() # optimise parameters
    # reset gradients to None instead of zero; this saves memory without altering computation
    optimiser.zero_grad(set_to_none =True)
    scheduler.step(cost) # update scheduler, tracks cost and updates learning rate if on plateau   

    network.train(False) # set network out of training mode
    return cost.detach().cpu().numpy() # store cost in a numpy array on cpu to allow plotting

def plotNetwork(network, mu, batchNum,
                xRange, yRange,uRange,vRange,tRange, numTimeSteps):
    x,y,u,v,t = DataSet(xRange,yRange,uRange,vRange,tRange,10).data_in
    batch = torch.cat((x,y,u,v,t),1)
    t = torch.linspace(tRange[0],tRange[1],numTimeSteps,requires_grad=True).view(-1,1)
    t = t.to(device)

    for i in range(len(batch)):
        x = torch.tensor([batch[i][0] for _ in range(numTimeSteps)]).view(-1,1)
        y = torch.tensor([batch[i][1] for _ in range(numTimeSteps)]).view(-1,1)
        u = torch.tensor([batch[i][2] for _ in range(numTimeSteps)]).view(-1,1)
        v = torch.tensor([batch[i][3] for _ in range(numTimeSteps)]).view(-1,1)
        x.requires_grad_()
        y.requires_grad_()
        u.requires_grad_()
        v.requires_grad_()
        x = x.to(device)
        y = y.to(device)
        u = u.to(device)
        v = v.to(device)
        input = torch.cat((x,y,u,v,t),dim=1)
        # for j in range(5):
        #     print(input[j])
        # print(input)
        out = network(input)
        xOut, yOut, uOut, vOut = torch.split(out, 1, dim = 1)

        xTrial = trialSolution(x,xOut,t).detach().cpu().numpy()
        #print(len(xTrial))
        yTrial = trialSolution(y,yOut,t).detach().cpu().numpy()
        # print(xTrial)
        # print(yTrial)
        
        # Plot Runge-Kutta solution
        x0 = batch[i][0].item()
        y0 = batch[i][1].item()
        u0 = batch[i][2].item()
        v0 = batch[i][3].item()
        xExact, yExact = rungeKutta(x0, y0, u0, v0, tRange[0], mu, tRange[1], numTimeSteps)
        if i == len(batch)-1:
            plt.plot(xExact, yExact, color = 'b', label = "Runge-Kutta Solution")
            plt.plot(xTrial,yTrial, color = 'r', label = "Neural Network Output")
        else:
            plt.plot(xExact, yExact, color = 'b')
            plt.plot(xTrial,yTrial, color = 'r')

    plt.plot([0.],[0.], marker = '.', markersize = 40)
    plt.plot([1.],[0.], marker = '.', markersize = 10)
    plt.xlabel('x', fontsize = 16)
    plt.ylabel('y', fontsize = 16)
    plt.legend(loc = "lower right", fontsize = 14)
    plt.title(str(batchNum) + " Batches", fontsize = 16)
    plt.show()

def dxdt(x,y,u,v,mu):
    """
    Returns RHS of differential equation for x'(t) at time t
    Arguments:
    x (float) -- value of x at time t
    y (float) -- value of y at time t
    u (float) -- value of u at time t
    v (float) -- value of v at time t
    Returns:
    x'(t) (float) -- value of x'(t) at time t
    """
    return u

def dydt(x,y,u,v,mu):
    """
    Returns RHS of differential equation for y'(t) 
    Arguments:
    x (float) -- value of x at time t
    y (float) -- value of y at time t
    u (float) -- value of u at time t
    v (float) -- value of v at time t
    Returns:
    y'(t) (float) -- value of y'(t) at time t
    """
    return v

def dudt(x,y,u,v,mu):
    """
    Returns RHS of differential equation for u'(t)
    Arguments:
    x (float) -- value of x at time t
    y (float) -- value of y at time t
    u (float) -- value of u at time t
    v (float) -- value of v at time t
    Returns:
    u'(t) (float) -- value of u'(t) at time t
    """
    return (x - mu + 2*v - (((mu*(x-1)) / ((x-1)**2 + y**2)**(3/2)) 
                + ((1-mu)*x / (x**2 + y**2)**(3/2))))

def dvdt(x,y,u,v,mu):
    """
    Returns RHS of differential equation for v'(t)
    Arguments:
    x (float) -- value of x at time t
    y (float) -- value of y at time t
    u (float) -- value of u at time t
    v (float) -- value of v at time t
    Returns:
    v'(t) (float) -- value of v'(t) at time t
    """
    return (y - 2*u - (((mu * y) / ((x-1)**2 + y**2)**(3/2))
                +((1-mu)*y / (x**2 + y**2)**(3/2) )) )

def rungeKutta(x0, y0, u0, v0, t0, mu, tFinal, numTimeSteps):
    """
    Implements 4th-Order Runge-Kutta Method to evaluate system of ODEs from time t0 to time tFinal
    Arguments:
    x0 (float) -- initial value of x at time t0
    y0 (float) -- initial value of y at time t0
    u0 (float) -- initial value of u at time t0
    v0 (float) -- initial value of v at time t0
    t0 (float) -- initial time value
    mu (float) -- non-dimensionalised mass of second body
    tFinal (float) -- final time value
    numTimeSteps (int) -- number of time step evaluations between t0 and tFinal
    Returns:
    xList (list of length numTimeSteps) -- x-values at each time step value
    yList (list of length numTimeSteps) -- y-values at each time step value
    """
    # Find size of time step by dividing interval width by numTimeSteps
    timeStepSize = int((tFinal - t0)/numTimeSteps)

    x, y, u, v, t = x0, y0, u0, v0, t0
    xList, yList = [x0], [y0]
    for _ in range(numTimeSteps):  # Iterate for number of time steps
        # Apply Runge Kutta Formulas to find next values of x, y, u, v
        a1 = timeStepSize * dxdt(x,y,u,v,mu)
        b1 = timeStepSize * dydt(x,y,u,v,mu)
        c1 = timeStepSize * dudt(x,y,u,v,mu)
        d1 = timeStepSize * dvdt(x,y,u,v,mu)
        
        a2 = timeStepSize * dxdt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)
        b2 = timeStepSize * dydt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)
        c2 = timeStepSize * dudt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)        
        d2 = timeStepSize * dvdt(x + 0.5*a1, y + 0.5*b1, u + 0.5*c1, v + 0.5*d1, mu)

        a3 = timeStepSize * dxdt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)
        b3 = timeStepSize * dydt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)
        c3 = timeStepSize * dudt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)        
        d3 = timeStepSize * dvdt(x + 0.5*a2, y + 0.5*b2, u + 0.5*c2, v + 0.5*d2, mu)

        a4 = timeStepSize * dxdt(x + a3, y + b3, u + c3, v + d3, mu)
        b4 = timeStepSize * dydt(x + a3, y + b3, u + c3, v + d3, mu)
        c4 = timeStepSize * dudt(x + a3, y + b3, u + c3, v + d3, mu)        
        d4 = timeStepSize * dvdt(x + a3, y + b3, u + c3, v + d3, mu)

        # Update next value of x, y, u, v
        x += (1.0 / 6.0)*(a1 + 2 * a2 + 2 * a3 + a4)
        y += (1.0 / 6.0)*(b1 + 2 * b2 + 2 * b3 + b4)
        u += (1.0 / 6.0)*(c1 + 2 * c2 + 2 * c3 + c4)
        v += (1.0 / 6.0)*(d1 + 2 * d2 + 2 * d3 + d4)

        # Store x- and y-values
        xList.append(x)
        yList.append(y)
        # Update next value of t
        t += timeStepSize
    return xList, yList


if torch.cuda.is_available():
    print("GPU available")
    device=torch.device("cuda")
else:
    print("no GPU available")
    device=torch.device("cpu")

xRange = [1.05,1.052]
yRange = [0.099, 0.101]
uRange = [-0.5,-0.4]
vRange = [-0.3,-0.2]
tRange = [-0.01,3]
batchSize = 10
mu = 0.01
lmbda = 2
numTimeSteps = 1000
numTotalBatches = 3000000

try: # load model if possible
    checkpoint  = torch.load('threeBodyOriginalMethod.pth')
    batchNum    = checkpoint['batchNum']
    network     = checkpoint['network']
    optimiser   = checkpoint['optimiser']
    scheduler   = checkpoint['scheduler']
    costs       = checkpoint['costs']
    print("model loaded")
except:
    try: # load initial state of model
        checkpoint = torch.load('threeBodyInitialNetwork.pth')
        network = checkpoint['network']
        print("initial model loaded")
    except:  # create new model and save its initial state
        network     = SolutionBundle(numHiddenNodes=128, numHiddenLayers=8)
        checkpoint  = {'network' : network}
        torch.save(checkpoint, 'threeBodyInitialNetwork.pth')
        print("new model created")
    batchNum = 0
    optimiser = torch.optim.Adam(network.parameters(), lr = 1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, factor = 0.5, patience = 200000, 
                    threshold = 0.5, min_lr = 1e-6, verbose = True)
    costs = []
network = network.to(device) # move network to GPU if available
lossFn  = torch.nn.MSELoss()

while batchNum <= numTotalBatches:
    newCost = train(network, lossFn, optimiser, scheduler, xRange,
                    yRange, uRange, vRange, tRange, batchSize, mu, lmbda)
    costs.append(newCost)
    if batchNum % 50000 == 0 : # save network every 50000 batches
        plotNetwork(network, mu, batchNum,
            xRange, yRange,uRange,vRange,tRange, numTimeSteps)
        checkpoint = {'batchNum': batchNum, 'network': network, 'optimiser': optimiser,
                        'scheduler': scheduler, 'costs': costs}
        torch.save(checkpoint, 'threeBodyOriginalMethod.pth')
        print("model saved")
    batchNum += 1

while batchNum <= numTotalBatches:
    # train on different curriculum depending on current batch number
    if batchNum < int(numTotalBatches/4):
        tRange = [-0.01,1]
    elif batchNum < int(numTotalBatches/2):
        tRange = [1,2]
    elif batchNum < int(3*numTotalBatches/4):
        tRange = [2,3]
    else:
        tRange = [-0.01,3]
    newCost = train(network, lossFn, optimiser, scheduler, xRange,
                    yRange, uRange, vRange, tRange, batchSize, mu, lmbda)
    costs.append(newCost)
    if batchNum % 50000 == 0 : # save network every 50000 batches
        plotNetwork(network, mu, batchNum,
            xRange, yRange,uRange,vRange,tRange, numTimeSteps)
        checkpoint = {'batchNum': batchNum, 'network': network, 'optimiser': optimiser,
                        'scheduler': scheduler, 'costs': costs}
        torch.save(checkpoint, 'threeBodyOriginalMethod.pth')
        print("model saved")
    batchNum += 1

while batchNum <= numTotalBatches:
    # widen time interval based n current batch number
    tFinal = min(3, np.exp( (3 * np.log(6) * batchNum) / (2.5 * numTotalBatches)) / 2)
    tRange = [-0.01, tFinal]
    newCost = train(network, lossFn, optimiser, scheduler, xRange,
                    yRange, uRange, vRange, tRange, batchSize, mu, lmbda)
    costs.append(newCost)
    if batchNum % 50000 == 0 : # save network every 50000 batches
        plotNetwork(network, mu, batchNum,
            xRange, yRange,uRange,vRange,tRange, numTimeSteps)
        checkpoint = {'batchNum': batchNum, 'network': network, 'optimiser': optimiser,
                        'scheduler': scheduler, 'costs': costs}
        torch.save(checkpoint, 'threeBodyOriginalMethod.pth')
        print("model saved")
    batchNum += 1

print(f"{batchNum} batches total, final loss = {costs[-1]}")

# %%

