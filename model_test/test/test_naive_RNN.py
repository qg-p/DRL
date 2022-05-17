'''
测试 DRQN 的朴素 RNN
输入：0~9 的数字 n
输出：n+1
用位置表示
'''
# 简单的序列分类

import torch
from torch import nn
class naive_RNN(nn.Module):
	'''
	RNN:
		output: network(cat([input, hidden]))+input;
		hidden: output.detach();
	input:	Tensor([batch size, num channel])
	output:	Tensor([batch size, num channel])
	hidden:	Tensor([batch size, num channel])
	'''
	def __init__(self, network:nn.Module):
		'''
		network:
			输入 channel=2n，输出 channel=n
		'''
		super().__init__()
		self.network = network
	def forward(self, y:torch.Tensor, RNN_states:torch.Tensor):
		RNN_input = torch.cat((y, RNN_states), axis=-1) # [batch_size][n*2]
		RNN_output:torch.Tensor = self.network.forward(RNN_input) # [batch_size][n]
		y = RNN_output + y # 防止梯度消失或爆炸
		RNN_states = y.detach()
		return y, RNN_states
class Model(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.RNN1 = naive_RNN(
			nn.Sequential(
				nn.Linear(8, 8), nn.Sigmoid(),
				nn.Linear(8, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
			)
		)
		self.RNN2 = naive_RNN(
			nn.Sequential(
				nn.Linear(8, 8), nn.ReLU(),
				nn.Linear(8, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
				nn.Linear(4, 4), nn.ReLU(),
			)
		) # 重要：不要两个激活函数相接 # 虽然不知道为什么，但 RNN 最后一层不用激活函数效果似乎比较好
		self.hidden = nn.Sequential(
			nn.Linear(12, 12), nn.ReLU(),
			nn.Linear(12, 4), nn.ReLU(),
			nn.Linear(4, 4), nn.ReLU(),
			nn.Linear(4, 4),
		)
		self.func = nn.Sequential(
			nn.Linear(4, 4), nn.ReLU(),
			nn.Linear(4, 4), nn.ReLU(),
			nn.Linear(4, 8), nn.ReLU(),
			nn.Linear(8, 32), nn.Softmax(-1),
			nn.Linear(32, 4)
		)
	from typing import List
	def forward(self, x:List[torch.Tensor]):
		s = torch.zeros([1, 4])
		for i in x:
			x = i.unsqueeze(0)
			S = s
			y, s = self.RNN1(x, s)#self.RNN2(*)
			s:torch.Tensor = self.hidden(torch.cat((S, x, s), axis=-1))
			s = s + S + y
			#s = s.detach()
		y:torch.Tensor = self.func(y)
		return y.flatten()
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
# train
inputs = [
	[[1], [2], [3], [0]],
	[[1], [0]],
	[[2], [3], [0]],
	[[0], [3], [0], [1], [0]],
]
for i in inputs:
	for j, J in enumerate(i):
		i[j] = torch.tensor([0, 0, 0, 0], dtype=torch.float)
		i[j][J[0]] = 1.
# for i in inputs: print(i)
labels = [
	torch.tensor([1, 0, 0, 0], dtype=torch.float),
	torch.tensor([0, 1, 0, 0], dtype=torch.float),
	torch.tensor([0, 0, 1, 0], dtype=torch.float),
	torch.tensor([0, 0, 0, 1], dtype=torch.float),
]
# print(inputs, labels)
losses = []
for i in range(300):
	print('epoch %d'%(i), end='\r')
	for x, y in zip(inputs, labels):
		predict_y = model.forward(x)
		labeled_y = y
		loss:torch.Tensor = loss_func(predict_y, labeled_y)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		losses.append(loss.item())
print('predict | label')
for x, y in zip(inputs, labels):
	predict_y = model.forward(x).data
	labeled_y = y
	print(predict_y, predict_y.argmax().item(), labeled_y.argmax().item())
from matplotlib import pyplot as plt
from math import log
plt.plot([log(i) for i in losses], marker='.', markersize=2, linewidth=0)
plt.show()