import torch
from torch import nn
loss_func = nn.SmoothL1Loss(reduce=True)

model = nn.Sequential(
	nn.Linear(1, 4), nn.Tanh(),
	nn.Linear(4, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 8), nn.ReLU(),
	nn.Linear(8, 1),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

def func(x):
	return 0.01*x**4 - 2*x**3 + 0.9*x**2 + 4*x - 1

l, r, d = -2, 2, 0.1
x = torch.arange(l+d/2, r+d/2, d).unsqueeze(1)
label = func(x)

losses = [0.]*0

for i in range(500):
	print('epoch %d'%(i), end='\r')
	predict:torch.Tensor = model(x)
	loss_vector:torch.Tensor = loss_func(predict, label)
	optimizer.zero_grad()
	loss_vector.backward()
	optimizer.step()
	losses.append(loss_vector.data.mean())
print()

predict = model(x).detach()
x, label, predict = x.flatten(), label.flatten(), predict.flatten()
from matplotlib import pyplot as plt
plt.plot(losses)
plt.figure()
plt.plot(x, label)
plt.plot(x, predict, marker='.', linewidth=0)
plt.show()