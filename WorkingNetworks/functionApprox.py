#%%
import matplotlib.pyplot as plt
import torch

class DataSet(torch.utils.data.Dataset):
    """
    An object which generates the x values for the input node 
    and the corresponding true values y=f(x) of the output node.
    """
    def __init__(self, fcn, xRange, numSamples):
        """
        Arguments:
        fcn (function) -- function to be approximated
        xRange (list of length 2) -- lower and upper limits for input values x
        numSamples (int) -- number of training data samples

        Returns:
        DataSet object with two attributes:
            dataIn (PyTorch tensor of shape (numSamples,1)) -- 'numSamples'
                evenly-spaced data points from xRange[0] to xRange[1]
            dataOut (PyTorch tensor of shape (numSamples,1)) -- corresponding 
                values of function at these points
        """
        self.dataIn  = torch.linspace(xRange[0], xRange[1], numSamples).view(-1,1)
        self.dataOut = fcn(self.dataIn).view(-1,1)
        # 'view' method reshapes tensors, in this case into column vectors

    def __len__(self):
        """
        Arguments:
        None

        Returns:
        len(self.dataIn) (int) -- number of training data points
        """
        return len(self.dataIn)
        
    def __getitem__(self, idx):
        """
        Used by DataLoader object to retrieve training data points

        Arguments:
        idx (int) -- index of data point required

        Returns:
        (x,f(x)) (tuple of 1x1 tensors) -- (x,f(x)) pair at index idx
        """
        return (self.dataIn[idx], self.dataOut[idx])
    
    
class Fitter(torch.nn.Module):
    """
    The neural network object, with 1 node in the input layer,
    1 node in the output layer, and 1 hidden layer with 'numHiddenNodes' nodes.
    """
    def __init__(self, numHiddenNodes):
        """
        Arguments:
        numHiddenNodes (int) -- number of nodes in hidden layer

        Returns:
        Fitter object (neural network) with two attributes:
        fc1 (fully connected layer) -- linear transformation of hidden layer
        fc2 (fully connected layer) -- linear transformation of outer layer
        """
        super(Fitter, self).__init__()
        self.fc1 = torch.nn.Linear(in_features = 1, out_features = numHiddenNodes)
        self.fc2 = torch.nn.Linear(in_features = numHiddenNodes, out_features = 1)

    def forward(self, x):
        """
        Function which connects inputs to outputs in the neural network.

        Arguments:
        x (PyTorch tensor shape (batchSize,1)) -- input of neural network

        Returns:
        y (PyTorch tensor shape (batchSize,1)) -- output of neural network
        """
        # tanh activation function used on hidden layer
        h = torch.tanh(self.fc1(x))
        print(h.shape)
        # Linear activation function used on outer layer
        y = self.fc2(h)
        return y
    
def train(network, loader, lossFn, optimiser, numEpochs):
    """
    A function to train a network.

    Arguments:
    network (Module) -- the neural network
    loader (DataLoader) -- generates batches from the training dataset
    lossFn (Loss Function) -- network's loss function
    optimiser (Optimiser) -- carries out parameter optimisation
    numEpochs (int) -- number of training epochs

    Returns:
    costList (list of length 'numEpochs') -- cost values of all epochs
    """
    costList = []
    network.train(True) # set module in training mode
    for epoch in range(numEpochs):
        for batch in loader:
            x, y = batch[0], batch[1] # x and y=f(x) values
            yOut = network.forward(x) # network output, prediction of y
            cost  = lossFn(yOut, y) # calculate cost value
            cost.backward() # back propagation, calculate gradients
            optimiser.step() # perform parameter optimisation
            optimiser.zero_grad() # reset gradients to zero
        costList.append(cost.item()) # store cost of each epoch
    network.train(False) # set module out of training mode
    return costList

fcn         = torch.sin
numEpochs   = 10000

network     = Fitter(numHiddenNodes=16)
trainSet    = DataSet(fcn, xRange = [-3,3], numSamples=30)
loader      = torch.utils.data.DataLoader(dataset=trainSet, batch_size=30)
lossFn      = torch.nn.MSELoss() # mean-squared error loss
optimiser   = torch.optim.SGD(network.parameters(), lr=1e-2)
# gradient descent algorithm (batch_size in loader determines 
# type (i.e. batch, mini-batch, stochastic))

costList    = train(network, loader, lossFn, optimiser, numEpochs)

x       =   torch.linspace(-3, 3, 30).view(-1,1)
yExact  =   fcn(x)
yOut    =   network.forward(x).detach().numpy() # tensor -> numpy array
x       =   x.detach().numpy() 
plt.plot(x, yExact,'b.', label = 'Exact Solution y = sin(x)')
plt.plot(x,yOut,'r-', label = 'Neural Network Output')
plt.xlabel('x', fontsize =16)
plt.ylabel('y', fontsize =16)
plt.title('Tanh Activation Function', fontsize=16)
plt.legend(loc = 'upper left',  fontsize=13)
plt.show()

plt.semilogy(costList)
plt.xlabel('Epoch', fontsize =16)
plt.ylabel('Cost', fontsize =16)
plt.title('Network Cost', fontsize=16)
plt.show()

print("Final cost = ", costList[-1])

"""
Effect of learning rate on final loss (with sine function, 2000 epochs)
Above 0.7 final loss increases rapidly
Between 0.1 and 0.25 final loss varies little, stays around 0.002
"""
# lossList = []
# for i in range(1,20):
#     network      = Fitter(numHiddenNodes=20)
#     trainSet    = DataSet(num_samples=20)
#     loader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=20, shuffle=True)
#     lossFn      = torch.nn.MSELoss()
#     optimiser    = torch.optim.SGD(network.parameters(), lr= 1e-1 + i * 1e-2)

#     loss = train(network, loader, lossFn, optimiser, epochs=2000)
#     lossList.append(loss)

# x = [1e-1 + i * 1e-2 for i in range(1,20)]
# plt.plot(x,lossList)
# plt.xlabel("Learning Rate")
# plt.ylabel("Final Loss")
# plt.show()


"""
Effect of number of epochs on final loss `ce function, learning rate = 1e-1
Sharp decrease from 1000 to 2000 (0.008 to 0.002)
Shallow decrease from 2000 to 4000 (0.002 to 0.001)
Fluctuations between 4000 and 9000 (between 0.0013 and 0.0008)

Fluctations between 4000 and 14000, but overall descent (from 0.0011 to 0.0006)
"""
# lossList = []
# for i in range(21):
#     network      = Fitter(numHiddenNodes=20)
#     trainSet    = DataSet(num_samples=20)
#     loader = torch.utils.data.DataLoader(dataset=trainSet, batch_size=20, shuffle=True)
#     lossFn      = torch.nn.MSELoss()
#     optimiser    = torch.optim.SGD(network.parameters(), lr= 1e-1)

#     loss = train(network, loader, lossFn, optimiser, epochs= 4000 + i *500)
#     lossList.append(loss)

# x = [4000 + i *500 for i in range(21)]
# print(lossList)
# plt.plot(x,lossList)
# plt.xlabel("Epochs")
# plt.ylabel("Final Loss")
# plt.show()
# %%

# %%

# %%

# %%
