import numpy as np

def sigmoid(x):
    return 1 / (1+ np.exp(-x))

sigmoid_vect = np.vectorize(sigmoid)
x = np.array([1,2,3,4])
#print(sigmoid_vect(x) + 7)

food = np.array([[56,0,4.4,68],
                [1.2,104,52,8],
                [1.8,135,99,0.9]])
assert(food.shape == (3,4))

cal = food.sum(axis = 0)
percentage = 100 * food/cal
print(percentage)