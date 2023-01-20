#%%

import torch
import time
import numpy
import math
import matplotlib
import matplotlib.pyplot as plt

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.fc1 = torch.nn.Linear(2,1)
        self.fc2 = torch.nn.Linear(1,2)

    def forward(self, p):
        a = torch.sigmoid(self.fc1(p))
        x = self.fc2(a) # no sigmoid on last layer, so it is linear in the middle node
        return x

    def from_funnel(self, a):
        # Ignore for now; this is a function to generate outputs by setting the
        # hidden node to a particular value.
        x = self.fc2(a)
        return x
    

class TrainDataSet(torch.utils.data.Dataset):
    def __init__(self, angle, samples):
        global device
        # Create a cloud of points centered around (0, 0).
        qq = numpy.random.rand(2, samples).astype('f') - 0.5
        # Squeeze along the y-direction more than the x-direction
        # to make a cloud along the x-axis.
        qq = numpy.array([[0.5], [0.1]]).astype('f')*qq
        # Rotate the whole cloud anti-clockwise over
        # an angle alpha.
        sa = math.sin(angle)
        ca = math.cos(angle)
        #print(math.atan2(ca, sa), "vs", angle)
        rot = numpy.array([[ca, -sa], [sa, ca]]).astype('f')
        # Shift to be centered at (1.0, 1.0)
        self.orig = 1.0 + numpy.matmul( rot, qq )
        self.data = torch.from_numpy(self.orig).to(device)

    def __getitem__(self, index):
        return self.data[:, index]

    def __len__(self):
        return len(self.data[0])
    

def train(model, loader, loss_fn, optimiser, epochs):
    global device
    losses = []
    model.train(True)
    for epoch in range(epochs):
        for v_in in loader:
            v_in.requires_grad=True
            v_out = model(v_in)
            loss  = loss_fn(v_out, v_in)
            loss.backward()
            # print(v_in.grad)
            optimiser.step()
            optimiser.zero_grad()
            losses.append(loss.item())
    model.train(False)
    print(" loss = ", loss.item())
    return losses

if torch.cuda.is_available():
    print("cuda time")
    device=torch.device("cuda")
else:
    print("sorry no cuda for yuda")
    device=torch.device("cpu")
    
alpha        = math.pi/4.0
model        = AutoEncoder()
train_set    = TrainDataSet(angle=alpha, samples=20)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=20, shuffle=True)
loss_fn      = torch.nn.MSELoss()
optimiser    = torch.optim.SGD(model.parameters(), lr=1e-2)
# optimiser    = torch.optim.Adam(model.parameters(), lr=1e-2)
model        = model.to(device)

# losses = train(model, train_loader, loss_fn, optimiser, epochs=4000)

# plt.plot( train_set.orig[0], train_set.orig[1], 'b.')
# plt.xlim(-2,2)
# plt.ylim(-2,2)
# plt.grid(True)
# xs=[]
# ys=[]
# for a in numpy.linspace(-20, 20, 20):
#    t = model.from_funnel( torch.Tensor([a]).to(device) ).detach().cpu().numpy()
#    xs.append(t[0])
#    ys.append(t[1])
   
# plt.plot(xs, ys, 'g-')
# plt.show()

losses=[1]
iterations=0
epochs=4000
while losses[-1]>0.001 and iterations < 10:
    iterations += 1
    losses.extend( train(model, train_loader, loss_fn, optimiser, epochs=epochs) )
losses = losses[1:]
print(f"{iterations*epochs} epochs total, final loss = {losses[-1]}")

plt.plot( train_set.orig[0], train_set.orig[1], 'b.', label = "Training Data")
plt.xlim(0,2)
plt.ylim(0,2)
plt.grid(True)
xs=[]
ys=[]
for a in numpy.linspace(-20, 20, 20):
   t = model.from_funnel( torch.Tensor([a]).to(device) ).detach().cpu().numpy()
   xs.append(t[0])
   ys.append(t[1])
   
plt.plot(xs, ys, 'g-', label = "Principal Component")
plt.legend(loc = "lower right")
plt.title("SGD")
plt.show()

plt.semilogy(losses)
plt.xlabel("Epochs")
plt.ylabel("Log of Loss")
plt.title("SGD")
# %%
