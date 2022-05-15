'''
LSTM 测试
输入：0~9 的数字 n
输出：n+1
用位置表示
'''
# 简单的序列分类

import torch
from torch import nn

class Model(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.LSTM = nn.LSTM(4, 4, batch_first=False)
		self.func = nn.Softmax(-1)
		self.fc = nn.Linear(4, 4)
	from typing import List
	def forward(self, x:List[torch.Tensor]):
		s = None
		for i in x:
			y, s = self.LSTM(i.view(1, 1, 4), s)
			# s = s[0].detach(), s[1].detach()
		y = self.func(y)
		y:torch.Tensor = self.fc(y)
		return y.flatten()
model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_func = nn.MSELoss()
# train
inputs = [
	[[1], [2], [3]],
	[[1], [0]],
	[[2], [3], [0]],
	[[0], [3], [0], [1]],
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
for i in range(500):
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