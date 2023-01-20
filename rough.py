#%%

import torch
# from torch.autograd import grad
# import torch.nn as nn
# import torch.optim as optim
import numpy as np
import scipy.io

data = scipy.io.loadmat('burgersData.mat')

t = torch.tensor(data['t'].flatten()[:,None])
x = torch.tensor(data['x'].flatten()[:,None])
Exact = np.real(data['usol']).T

X, T = np.meshgrid(x,t)

X = torch.tensor(X)
T = torch.tensor(T)

X_star = np.hstack((X.flatten()[:,None], T.flatten()[:,None]))
u_star = Exact.flatten()[:,None]   
print(t)
print(x)
print(Exact.shape)
print(X)
print(T)
print(X_star.shape)
print(u_star.shape)

idx = np.random.choice(X_star.shape[0], 2000, replace=False)
X_u_train = X_star[idx,:]
u_train = u_star[idx,:]
print(idx)
print(X_u_train.shape)
print(u_train.shape)
# # x = [[1,2,3],
# #      [4,5,6],
# #      [7,8,9]]

# # print([x[i][0] for i in range(3)])

# # amount = 4
# # dp = [amount +1] * (amount + 1)
# # print(dp)

# # x = torch.tensor([[1,2],
# #                   [3,4],
# #                   [5,6]])

# # x = x.detach().numpy()
# # print(x)

# # y = torch.tensor([[-1,1],
# #                   [3,5],
# #                   [7,9]])

# # y = y.detach().numpy()
# # print(y)

# # print(np.square(x-y).mean())


# # x = [[1,2,3],
# #      [4,5,6]]

# # y = np.array(x)
# # print(y.shape)
# # z = np.exp(y)
# # print(z)

# # coordinates = (0,0,2)
# # print(y[coordinates])

# # x = [0,1,2]
# # y = torch.tensor(x)
# # print(y.size(dim = 0))

# # def exp_reducer(x):
# #   return x.exp().sum(dim=0)
# # inputs = torch.rand(2, 5)
# # print(inputs)
# # print(inputs.exp())
# # print(exp_reducer(inputs))
# # print(jacobian(exp_reducer, inputs))

# # print(jacobian(exp_reducer, inputs).shape)

# # def exp_adder(x, y):
# #      # print(2* x.exp())
# #      # print(3 * y)

# #      return 2 * x.exp() + 3 * y
# # inputs = (torch.rand(2), torch.rand(2))
# # print(inputs)
# # print(exp_adder(inputs[0],inputs[1]))
# # jacobian(exp_adder, inputs)

# class net_x(nn.Module): 
#         def __init__(self):
#             super(net_x, self).__init__()
#             self.fc1=nn.Linear(2, 20) 
#             self.fc2=nn.Linear(20, 20)
#             self.out=nn.Linear(20, 4) #a,b,c,d

#         def forward(self, x):
#             x=torch.tanh(self.fc1(x))
#             x=torch.tanh(self.fc2(x))
#             x=self.out(x)
#             return x

# nx = net_x()

# #input

# val = 10
# a = torch.rand(val, requires_grad = True) #input vector
# print(a)
# t = torch.reshape(a, (5,2)) #reshape for batch
# print(t)

# # #method 
# # dx = torch.autograd.functional.jacobian(nx, t)
# # print(dx.shape)
# # print(dx)
# # print(torch.diagonal(dx,0,-1))
# # #dx = torch.diagonal(torch.diagonal(dx, 0, -1), 0)[0] #first vector
# # dx = torch.diagonal(torch.diagonal(dx, 1, -1), 0)[0] #2nd vector
# # #dx = torch.diagonal(torch.diagonal(dx, 2, -1), 0)[0] #3rd vector
# # #dx = torch.diagonal(torch.diagonal(dx, 3, -1), 0)[0] #4th vector
# # print(dx)

# print(a.grad)
# out = nx(t)
# m = torch.zeros((5,4))
# print(m)
# m[:, 0] = 1
# print(m)
# out.backward(m)
# print(a.grad)


# %%
