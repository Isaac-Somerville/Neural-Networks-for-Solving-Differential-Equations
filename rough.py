import torch
import time
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

# x = [[1,2,3],
#      [4,5,6],
#      [7,8,9]]

# print([x[i][0] for i in range(3)])

# amount = 4
# dp = [amount +1] * (amount + 1)
# print(dp)

# x = torch.tensor([[1,2],
#                   [3,4],
#                   [5,6]])

# x = x.detach().numpy()
# print(x)

# y = torch.tensor([[-1,1],
#                   [3,5],
#                   [7,9]])

# y = y.detach().numpy()
# print(y)

# print(np.square(x-y).mean())


x = [[1,2,3],
     [4,5,6]]

y = np.array(x)
print(y.shape)
z = np.exp(y)
print(z)

# coordinates = (0,0,2)
# print(y[coordinates])

# x = [0,1,2]
# y = torch.tensor(x)
# print(y.size(dim = 0))

