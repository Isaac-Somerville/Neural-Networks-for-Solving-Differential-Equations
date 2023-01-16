import numpy as np
import matplotlib.pyplot as plt
import torch

class DataSet(torch.utils.data.Dataset):
    """
    An object which can generate the values $x$ for the input node 
    and the aimed-for corresponding values $f(x)$ of the output node.
    
    It's very simple here: x ranges from [-2,2], f(x) is the 'sin'
    function (or anything else you pass through the init function).
    """
    def __init__(self, num_samples, fun=np.sin):
        self.data_in  = np.linspace(-2, 2, num_samples, dtype='f')
        self.data_out = fun(self.data_in)  

    def __len__(self):
        return len(self.data_in)
        
    def __getitem__(self, i):
        return (self.data_in[i], self.data_out[i])
    
    
class Fitter(torch.nn.Module):
    """
    The neural network object, with 1 node in the input layer,
    1 node in the output layer, and 'num_hidden_nodes' nodes
    in the hidden layer. 
    """
    def __init__(self, num_hidden_nodes):
        super(Fitter, self).__init__()
        # Fully connected layers are called 'Linear' layers
        # in PyTorch.
        self.fc1 = torch.nn.Linear(1, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, x):
        # The forward function is what connects the layers
        # together; note the 'sigmoid' for the hidden
        # layer nodes, but not for the output nodes.
        h = torch.sigmoid(self.fc1(x))
        y = self.fc2(h)
        return y
    
def train(network, loader, loss_fn, optimiser, epochs):
    """
    A function to train a network on the data produced
    by the `loader` object.
    """
    network.train(True) #set module in training mode
    for epoch in range(epochs):
        for batch in loader:
            # Ensure that the input and output are tensors
            # for one node; anything larger implies that the
            # tensors contain a batch, not a single value.
            batch[0]=batch[0].view(-1,1) #x-values
            batch[1]=batch[1].view(-1,1) #true f(x) values
            v_out = network(batch[0]) # ?? forward propagation ??
            loss  = loss_fn(v_out, batch[1]) #calculate MSE loss
            loss.backward() #back propagation, calculates grads
            if epoch % 100 == 0:
                print(loss)
            optimiser.step() #gradient descent, updates params
            optimiser.zero_grad() #sets grads to zero
    network.train(False) #set module out of training mode
    x = loss.item()
    return x
    
network      = Fitter(num_hidden_nodes=20)
train_set    = DataSet(num_samples=20)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
loss_fn      = torch.nn.MSELoss()
optimiser    = torch.optim.SGD(network.parameters(), lr=1e-1)

train(network, train_loader, loss_fn, optimiser, epochs=2000)

x=torch.Tensor(np.linspace(-2, 2, 20)).view(-1,1)
t=np.sin(x)
y=network.forward(x).detach().numpy() #tensor -> numpy array
plt.plot(x,t,'b.')
plt.plot(x,y,'r-')
plt.show()

"""
Effect of learning rate on final loss (with sine function, 2000 epochs)
Above 0.7 final loss increases rapidly
Between 0.1 and 0.25 final loss varies little, stays around 0.002
"""
# lossList = []
# for i in range(1,20):
#     network      = Fitter(num_hidden_nodes=20)
#     train_set    = DataSet(num_samples=20)
#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
#     loss_fn      = torch.nn.MSELoss()
#     optimiser    = torch.optim.SGD(network.parameters(), lr= 1e-1 + i * 1e-2)

#     loss = train(network, train_loader, loss_fn, optimiser, epochs=2000)
#     lossList.append(loss)

# x = [1e-1 + i * 1e-2 for i in range(1,20)]
# plt.plot(x,lossList)
# plt.xlabel("Learning Rate")
# plt.ylabel("Final Loss")
# plt.show()


"""
Effect of number of epochs on final loss (since function, learning rate = 1e-1
Sharp decrease from 1000 to 2000 (0.008 to 0.002)
Shallow decrease from 2000 to 4000 (0.002 to 0.001)
Fluctuations between 4000 and 9000 (between 0.0013 and 0.0008)

Fluctations between 4000 and 14000, but overall descent (from 0.0011 to 0.0006)
"""
# lossList = []
# for i in range(21):
#     network      = Fitter(num_hidden_nodes=20)
#     train_set    = DataSet(num_samples=20)
#     train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
#     loss_fn      = torch.nn.MSELoss()
#     optimiser    = torch.optim.SGD(network.parameters(), lr= 1e-1)

#     loss = train(network, train_loader, loss_fn, optimiser, epochs= 4000 + i *500)
#     lossList.append(loss)

# x = [4000 + i *500 for i in range(21)]
# print(lossList)
# plt.plot(x,lossList)
# plt.xlabel("Epochs")
# plt.ylabel("Final Loss")
# plt.show()