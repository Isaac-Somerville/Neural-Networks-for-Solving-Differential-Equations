import torch
import time
import numpy
import math
import matplotlib
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(2,1) # 2 inputs, 1 hidden node
        self.fc2 = torch.nn.Linear(1,2) # 1 hidden node, 2 outputs

    def forward(self, p):
        a = torch.sigmoid(self.fc1(p))
        x = self.fc2(a) # no sigmoid on last layer, 
                        # so it is linear in the middle node
        return x

    def from_funnel(self, a):
        # Ignore for now; this is a function to 
        # generate outputs by setting the
        # hidden node to a particular value.
        x = self.fc2(a)
        return x
    
class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, angle, samples):
        """_summary_

        Args:
            angle (float): angle by which we rotate cloud of points
            samples (int): number of samples in cloud of points
        
        """
        global device
        # Create a cloud of points centered around (0, 0).
        qq = numpy.random.rand(2, samples).astype('f') - 0.5 
        # Squeeze along the y-direction more than the x-direction
        # to make a cloud along the x-axis
        qq = numpy.array([[0.5], [0.1]]).astype('f')*qq
        # Rotate the whole cloud anti-clockwise over
        # an angle alpha.
        sa = math.sin(angle)
        ca = math.cos(angle)
        print(math.atan2(sa, ca), "vs", angle)
        rot = numpy.array([[ca, -sa], [sa, ca]]).astype('f')
        # Shift to be centered at (1.0, 1.0)
        self.orig = 1.0 + numpy.matmul( rot, qq )
        self.data = torch.from_numpy(self.orig).to(device) #store tensor in device (cuda or cpu)

    def __getitem__(self, index):
        return self.data[:, index]

    def __len__(self):
        return len(self.data[0])
    
def train(loader, train_data, epochs):
    global device
    count = 0
    bestLineFound = False
    train_data = train_data.data.detach().numpy()
    trainData = sorted(train_data, key = lambda x : x[0])
    print(trainData)
    while count < 10  and bestLineFound == False:
        model = AutoEncoder()
        loss_fn      = torch.nn.MSELoss()
        optimiser    = torch.optim.SGD(model.parameters(), lr=1e-3)
        model        = model.to(device)
        losses = []
        model.train(True)
        for epoch in range(epochs):
            for v_in in loader:
                v_in.requires_grad=True #grads need to be computed
                v_out = model(v_in)
                loss  = loss_fn(v_out, v_in)
                loss.backward() #back propagation, calculates grads
                # print(v_in.grad)
                optimiser.step() #gradient descent, updates params
                optimiser.zero_grad()  #sets grads to zero
                losses.append(loss.item())
                if epoch % 100 == 0:
                    print(loss.item())
        model.train(False)
        testData = [[],[]]
        for a in numpy.linspace(0, 1, 20):
            t = model.from_funnel( torch.Tensor([a]).to(device) ).detach().cpu().numpy()
            testData[0].append(t[0])
            testData[1].append(t[1])
        testData = numpy.array(testData)
        print(numpy.square(trainData - testData).mean())
        if numpy.square(trainData - testData).mean() < 1e-2:
            bestLineFound = True
        count += 1
    return losses

if torch.cuda.is_available() and False:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

alpha        = math.pi/4
model        = AutoEncoder()
train_set    = TrainDataSet(angle=alpha, samples=20)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
loss_fn      = torch.nn.MSELoss()
optimiser    = torch.optim.SGD(model.parameters(), lr=1e-3)
model        = model.to(device)

losses = train(train_loader, train_set, epochs=2000)

plt.plot( train_set.orig[0], train_set.orig[1], 'b.')
plt.xlim(-2,2)
plt.ylim(-2,2)
plt.grid(True)
xs=[]
ys=[]
for a in numpy.linspace(0, 1, 20):
   t = model.from_funnel( torch.Tensor([a]).to(device) ).detach().cpu().numpy() 
   # CUDA tensor -> CPU tensor -> numpy array
   xs.append(t[0])
   ys.append(t[1])
   
plt.plot(xs, ys, 'g-')
plt.show()