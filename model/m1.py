if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
import nle_win as nle
'''
搭建测试用模型
'''
from replay_memory import replay_memory
from explore.glyphs import translate_glyphs, translate_glyph

import torch
from torch import nn
class Bottleneck(nn.Module):
	def __init__(self, layers:list=None) -> None:
		super().__init__()
		from typing import List
		self.layers:List[Conv] = layers if layers is not None else []

	def init(self):
		in_channels = self.layers[0].in_channels
		out_channels = self.layers[len(self.layers)-1].out_channels
		self.resample = [] # will not be used if is not top-level
		if in_channels != out_channels: # resample via conv 1*1
			self.resample.append(nn.Conv2d(in_channels, out_channels, 1, groups=in_channels))
		if any([layer.bn is not None for layer in self.layers]): # do batch-norm if anyone used
			self.resample.append(nn.BatchNorm2d(out_channels))

	def __or__(self, rhv):
		retv = Bottleneck(self.layers + rhv.layers)
		retv.init()
		return retv
	def forward(self, x):
		y = x
		for layer in self.layers:
			y = layer(y)

		res = x # residual
		for resample in self.resample:
			res = resample(res) # resampled residual

		y += res # bottleneck
		return y
class Conv(Bottleneck):
	# auto padding
	def __init__(self, in_channels:int, out_channels:int, kernel_size:int, fn, groups:int=1, bn:bool=False) -> None:
		super().__init__([self]) # register
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, groups=groups)
		self.bn = nn.BatchNorm2d(out_channels) if bn else None
		self.fn = fn
		self.network = nn.Sequential(self.conv, self.bn, self.fn) if self.bn else nn.Sequential(self.conv, self.fn)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.init()

	def forward(self, x):
		y = self.network(x)
		return y
class DQN(nn.Module):
	def __init__(self) -> None:
		super().__init__()
		self.Q = [ # List 表示顺序，Tuple 表示并列
			(
				[ # seq, process map 21 * 79
					nn.Sequential(
						nn.Conv2d(4, 4, 1), nn.Sigmoid(), # all grids are considered same
						Conv(4, 4, 1, nn.ReLU(inplace=True), 1, False) | Conv(4, 4, 1, nn.ReLU(inplace=True), 4, False),
						Conv(4, 4, 1, nn.ReLU(inplace=True), 1, False) | Conv(4, 4, 1, nn.ReLU(inplace=True), 4, True),
					), # resnet, preprocess
					nn.Sequential(
						nn.Conv2d(4, 1, 3, padding=1), nn.ReLU(inplace=True),
						nn.AvgPool2d(5),
						nn.Conv2d(1, 1, 3, padding=1), nn.ReLU(inplace=True),
						nn.BatchNorm2d(1)
					), # normal(?) CNN
				],
				nn.Sequential( # process map 5 * 5
					nn.Conv2d(4, 8, 3, padding=1), nn.Sigmoid(), nn.BatchNorm2d(8),
					Conv(8, 8, 3, nn.ReLU(inplace=True), 1, False) | Conv(8, 8, 3, nn.ReLU(inplace=True), 1, True),
					nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid(), nn.BatchNorm2d(1),
					Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, True),
					Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, True),
					Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, False) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1, True),
				),
				nn.Sequential( # BLSTATS MLP
					nn.Linear(26, 26), nn.BatchNorm1d(26), nn.Sigmoid(),
					nn.Linear(26, 8), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
					nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
					nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
					nn.Linear(8, 8), nn.BatchNorm1d(8), nn.ReLU(inplace=True),
					nn.Linear(8, 8), nn.BatchNorm1d(8)
				),
			),
			(
				nn.Sequential(
					nn.Linear(1, 1)
				)
			)
		]
	default_glyph = 2359 # ' '
	default_translation = translate_glyph(default_glyph)
	from typing import List
	def forward(self, obs_batch:List[nle.basic.obs.observation]):
		''' output: Q(obs)[len(actions)] '''
		from typing import List
		def preprocess(obs_batch:List[nle.basic.obs.observation]):
			map_batch = []
			surroundings_batch = []
			blstats_batch = []
			for obs in obs_batch:
				blstats_batch.append(obs.blstats)
				map_4chnl = translate_glyphs(obs) # map in 4 channels
				map_batch.append(map_4chnl.tolist())
				srdng5x5_4chnl = [[[int(DQN.default_translation[0])]*5]*5, [[int(DQN.default_translation[1])]*5]*5, [[int(DQN.default_translation[2])]*5]*5, [[int(DQN.default_translation[3])]*5]*5]
				y, x = obs.tty_cursor; y -= 1
				for i in range(max(0, y-2), min(map_4chnl.shape[1], y+3)): # shape: (4, 21, 79,)
					for j in range(max(0, x-2), min(map_4chnl.shape[2], x+3)):
						srdng5x5_4chnl[0][i-(y-2)][j-(x-2)] = int(map_4chnl[0][i][j])
						srdng5x5_4chnl[1][i-(y-2)][j-(x-2)] = int(map_4chnl[1][i][j])
						srdng5x5_4chnl[2][i-(y-2)][j-(x-2)] = int(map_4chnl[2][i][j])
						srdng5x5_4chnl[3][i-(y-2)][j-(x-2)] = int(map_4chnl[3][i][j])
				surroundings_batch.append(srdng5x5_4chnl)
			return torch.tensor(map_batch, dtype=torch.float), torch.tensor(surroundings_batch, dtype=torch.float), torch.tensor(blstats_batch, dtype=torch.float)
		map_4, srdng5x5_4, blstat_26 = preprocess(obs_batch)

		a = self.Q[0][0][0](map_4)
		a = self.Q[0][0][1](a)
		b = self.Q[0][1](srdng5x5_4)
		c = self.Q[0][2](blstat_26)

		return a, b, c

actions=[
	None, True, False, # q(uit), y(es), n(o)
	' ', # space, numbers, symbols and letters, return non-zero reward only-if misc[0] and not misc[1]
	'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
	'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
	# normal actions
	0, 1, 2, 3, 4, 5, 6, 7, # move
	16, 17, 18, # up, down, wait
	19, # more
	30, # close (door)
	38, # esc
	39, # fight
	48, # kick
	57, # open (door)
	61, # pick up
	75, # search
]

def __main__():
	nle.connect()
	policy_net = DQN()
	target_net = DQN() # 提升训练稳定性，但为何不用 DDQN
	policy_net.train()
	target_net.eval()

	memory = replay_memory(500)
	batch_size = 10
	import random

	done = True
	for n_ep in range(15):
		if done:
			nle.Exec('env.reset()')
			obs = nle.getobs()
		if n_ep > batch_size:
			if n_ep % 5 == 0:
				target_net.load_state_dict(policy_net.state_dict())
			batch = memory.sample(batch_size)
			training_data = []
			for i in batch:
				training_data.append(i.state)
			policy_net.forward(training_data)
		action = random.sample(actions[56:], 1)[0]
		last_obs = obs
		obs, reward, done = nle.step(action)
		memory.push(last_obs, action, reward, None if done else obs)
	nle.disconnect()
if __name__=='__main__':
	__main__()