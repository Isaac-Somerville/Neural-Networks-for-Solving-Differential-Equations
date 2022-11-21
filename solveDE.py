import torch
import time
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt


class DESolver(torch.nn.Module):
    def __init__(self, num_hidden_nodes):
        super(DESolver, self).__init__()
        self.fc1 = torch.nn.Linear(1, num_hidden_nodes)
        self.fc2 = torch.nn.Linear(num_hidden_nodes, 1)

    def forward(self, x):
        h = torch.sigmoid(self.fc1(x))
        y = self.fc2(h) 
        return y

class DataSet(torch.utils.data.Dataset):
    
    def __init__(self, num_samples, lower_lim, upper_lim, func):
        """
        Args: 
        
        num_samples (int) -- number of points in training set
        lower_lim (int) -- min value of training points
        upper_lim (int) -- max value of training points
        func (function) -- function x --> y 
        
        Returns:
        
        self.data_in (numpy array length num_samples) -- points in training set
        self.data_out (numpy array length num_samples) -- value of D(x) at points in training set
        """
        data = []
        for i in range(len(ranges)):
            next_axis = np.linspace(ranges[i][0], ranges[i][1], num_samples)
            data.append(next_axis)
        self.data_in = np.array(data)
        # TODO: self.data_out should be D(x) where D(x) = 0 is the DE
        self.data_out = np.ones(num_samples)
        
    def __len__(self):
        return self.data_in.shape
    
    def __getitem__(self, i):
        """
        Args:
        
        i (int) -- index of training point 
        
        Returns:
        input (numpy array dimension 1 * dim) -- ith training point
        output (float) -- value of D(x) at ith training point
        """
        input = []
        output = self.data_out[i]
        for j in range(self.data_in.shape[0]):
            input.append(self.data_in[j,i])
        return (input,output)

            

network = DESolver(dim = 1, num_hidden_nodes= 10)
train_set = DataSet(10,[[0,2]])
# print(train_set.data_in)
# print(train_set.data_out)
# print(train_set.__len__())
# print(train_set.__getitem__(4))
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=10, shuffle=True)
loss_fn = torch.nn.MSELoss()
optimiser = torch.optim.SGD(network.parameters(), lr=1e-1)




  
# def loss(nn_output, data_out):
#     """
#     Args:
#     nn_output (pytorch tensor length num_samples) -- output of neural network at training points
#     data_out (pytorch tensor length num_samples) -- values of D(x) at training points 
    
#     Returns:
#     loss (float) -- mean square difference of nn_output and data_out
#     """   
#     loss = torch.sum(torch.square(nn_output - data_out)) / data_out.size(dim=0)
#     return loss