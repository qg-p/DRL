if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys
from model.misc import *
from model.glyphs import translate_glyph, translate_glyphs, translate_inv, translate_messages_misc

import torch
from torch import nn

num_workers=8
from concurrent.futures import ThreadPoolExecutor
thread_pool_executor = ThreadPoolExecutor(max_workers=num_workers)

from typing import List
def preprocess(obs_batch:List[nle.basic.obs.observation], device:torch.device):
	def batch_split(n:int):
		L = len(obs_batch)
		r = [L//n]*n
		for i in range(L%n): r[i] += 1
		i, a = 0, [0]
		for r in r:
			i += r
			a.append(i)
		r = [obs_batch[i:j] for i, j in zip(a[:-1], a[1:])]
		return r
	if len(obs_batch)>=num_workers: # 否则效率较低
		works = [thread_pool_executor.submit(preprocess_single, split_batch) for split_batch in batch_split(num_workers)]
		results = [work.result() for work in works]
	else:
		results = [preprocess_single(obs_batch)]
	map_batch, surroundings_batch, blstats_batch, inv_batch, misc_batch = results[0]
	for map_batch_result, surroundings_batch_result, blstats_batch_result, inv_batch_result, misc_batch_result in results[1:]:
		map_batch += map_batch_result
		surroundings_batch += surroundings_batch_result
		blstats_batch += blstats_batch_result
		inv_batch += inv_batch_result
		misc_batch += misc_batch_result

	map_batch:torch.Tensor = torch.tensor(map_batch, dtype=torch.float, device=device)
	surroundings_batch:torch.Tensor = torch.tensor(surroundings_batch, dtype=torch.float, device=device)
	blstats_batch:torch.Tensor = torch.tensor(blstats_batch, dtype=torch.float, device=device)
	inv_batch:torch.Tensor = torch.tensor(inv_batch, dtype=torch.float, device=device)
	return map_batch, surroundings_batch, blstats_batch, inv_batch, misc_batch

def preprocess_single(obs_batch:List[nle.basic.obs.observation]):
	default_glyph = 2359 # ' '
	default_translation = translate_glyph(default_glyph)
	batch_size = len(obs_batch)
	map_batch = [None] * batch_size
	surroundings_batch = [None] * batch_size
	blstats_batch = [None] * batch_size
	misc_batch = [[0]] * batch_size
	inv_batch = [None] * batch_size

	for batch_i, obs in enumerate(obs_batch):
		misc_batch[batch_i] = translate_messages_misc(obs)
		inv_batch[batch_i] = translate_inv(obs)
		blstats_batch[batch_i] = obs.blstats
		map_4chnl = translate_glyphs(obs) # map in 4 channels
		map_batch[batch_i] = map_4chnl.tolist()
		srdng5x5_4chnl = [[[int(default_translation[0])]*5]*5, [[int(default_translation[1])]*5]*5, [[int(default_translation[2])]*5]*5, [[int(default_translation[3])]*5]*5]
		y, x = obs.tty_cursor; y -= 1
		for i in range(max(0, y-2), min(map_4chnl.shape[1], y+3)): # shape: (4, 21, 79,)
			for j in range(max(0, x-2), min(map_4chnl.shape[2], x+3)):
				srdng5x5_4chnl[0][i-(y-2)][j-(x-2)] = int(map_4chnl[0][i][j])
				srdng5x5_4chnl[1][i-(y-2)][j-(x-2)] = int(map_4chnl[1][i][j])
				srdng5x5_4chnl[2][i-(y-2)][j-(x-2)] = int(map_4chnl[2][i][j])
				srdng5x5_4chnl[3][i-(y-2)][j-(x-2)] = int(map_4chnl[3][i][j])
		surroundings_batch[batch_i] = srdng5x5_4chnl

	return map_batch, surroundings_batch, blstats_batch, inv_batch, misc_batch

class Bottleneck(nn.Module):
	def __init__(self, layers:list=None) -> None:
		super().__init__()
		from typing import List
		self.layers:List[Conv] = layers if layers is not None else []

	def init(self):
		in_channels = self.layers[0].in_channels
		out_channels = self.layers[len(self.layers)-1].out_channels
		resample = None # will not be used if is not top-level
		if in_channels != out_channels: # resample via conv 1*1
			resample = nn.Conv2d(in_channels, out_channels, 1) # , groups=in_channels
		self.resample = resample

		for no, layer in enumerate(self.layers):
			self.add_module('layer'+str(no), layer) # from nn.Sequential

		return self
	from typing_extensions import Self
	def __or__(self, rhv:Self):
		retv = Bottleneck(self.layers + rhv.layers)
		return retv
	def forward(self, x:torch.Tensor):
		res = x.clone() # residual
		if self.resample is not None:
			res = self.resample(res) # resampled residual

		y = x
		for layer in self.layers:
			y = layer(y)

		y = y + res # bottleneck
		return y
class Conv(Bottleneck):
	# auto padding
	def __init__(self, in_channels:int, out_channels:int, kernel_size:int, fn, groups:int=1) -> None:
		super().__init__([self]) # register
		conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, groups=groups)
		fn = fn
		self.network = nn.Sequential(conv, fn)
		self.in_channels = in_channels
		self.out_channels = out_channels

	def forward(self, x):
		y = self.network(x)
		return y
# class RNN(nn.Module):
# 	'''
# 	RNN:

# 		output = network(cat([input, hidden]))+input;

# 		hidden = output.detach();

# 	input:	Tensor([batch size, num channel])
# 	output:	Tensor([batch size, num channel])
# 	hidden:	Tensor([batch size, num channel])
# 	'''
# 	def __init__(self, network:nn.Module):
# 		'''
# 		network:
# 			输入 channel=2n，输出 channel=n
# 		'''
# 		super().__init__()
# 		self.network = network
# 	def forward(self, y:torch.Tensor, RNN_states:torch.Tensor):
# 		RNN_input = torch.cat((y, RNN_states), axis=-1) # [batch_size][128]
# 		RNN_output:torch.Tensor = self.network.forward(RNN_input) # [batch_size][64]
# 		y = RNN_output + y # 防止梯度消失或爆炸
# 		RNN_states = y.detach() # 将输出 y 作为 RNN 隐藏状态（一点也没有隐藏的意思）
# 		return y, RNN_states
# 	# INITIAL_RNN_STATE = torch.zeros(64)
class DRQN(nn.Module):
	def __init__(self, device, n_actions_ynq:int=len(actions_ynq), n_actions_normal:int=len(actions_normal)) -> None:
		super().__init__()
		self.device = device
		self.Q = [ # List 表示顺序，Tuple 表示并列
			(
				[ # seq, process map 4(channels) * 21 * 79
					nn.Sequential(
						nn.Conv2d(4, 64, 1), nn.Tanh(), # all grids are considered same
						(Conv(64, 64, 1, nn.ReLU(inplace=True), 1) | Conv(64, 64, 1, nn.ReLU(inplace=True), 16)).init(),
						(Conv(64, 64, 1, nn.ReLU(inplace=True), 1) | Conv(64, 64, 1, nn.ReLU(inplace=True), 16)).init(),
						(Conv(64, 64, 1, nn.ReLU(inplace=True), 1) | Conv(64, 16, 1, nn.ReLU(inplace=True), 16)).init(),
						(Conv(16, 16, 1, nn.ReLU(inplace=True), 1) | Conv(16, 16, 1, nn.ReLU(inplace=True), 16)).init(),
						(Conv(16, 16, 1, nn.ReLU(inplace=True), 1) | Conv(16, 16, 1, nn.ReLU(inplace=True), 16)).init(),
						(Conv(16, 16, 1, nn.ReLU(inplace=True), 1) | Conv(16, 4, 1, nn.ReLU(inplace=True), 4)).init(),
						(Conv(4, 4, 1, nn.ReLU(inplace=True), 1) | Conv(4, 4, 1, nn.ReLU(inplace=True), 4)).init(),
						(Conv(4, 4, 1, nn.ReLU(inplace=True), 1) | Conv(4, 4, 1, nn.ReLU(inplace=True), 4)).init(),
					), # resnet, preprocess
					nn.Sequential(
						nn.Conv2d(4, 1, 3, padding=1), nn.ReLU(inplace=True),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						# nn.MaxPool2d(3), # extract 'feature' such as corner, door, lava pool, etc
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						# nn.AvgPool2d(3), # downsample
						nn.Conv2d(1, 1, 3, padding=1), nn.ReLU(inplace=True),
						nn.Conv2d(1, 1, 3, padding=1),
					), # normal(?) CNN
				], # output: 1 * 21 * 79 ## output: 1 * 2 * 8
				[
					nn.Sequential( # process map 5 * 5
						nn.Conv2d(4, 8, 3, padding=1), nn.Tanh(),
						(Conv(8, 8, 3, nn.ReLU(inplace=True), 1) | Conv(8, 8, 3, nn.ReLU(inplace=True), 1)).init(),
						nn.Conv2d(8, 1, 3, padding=1), nn.Tanh(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
						(Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1) | Conv(1, 1, 3, nn.ReLU(inplace=True), 1)).init(),
					),
					nn.Sequential(
						nn.Conv2d(1, 1, 3), nn.ReLU(),
						nn.Flatten(1), nn.Linear(9, 9)
					)
				],
				nn.Sequential( # BLSTATS MLP
					nn.Linear(26, 26), nn.Tanh(), # abs. no bn.
					nn.Linear(26, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16),
				),
			),
			(
				nn.Sequential(
					nn.Linear(21*79+9+16, 64), nn.ReLU(inplace=True),
					nn.Linear(64, 64), nn.ReLU(inplace=True),
					nn.Linear(64, 64), nn.ReLU(inplace=True),
					nn.Linear(64, 64), nn.ReLU(inplace=True),
					nn.Linear(64, 64), nn.ReLU(inplace=True),
					nn.Linear(64, 64), nn.ReLU(inplace=True)
				),
				# [
				# 	RNN(nn.Sequential(
				# 		nn.Linear(128, 128), nn.Tanh(),
				# 		nn.Linear(128, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True)
				# 	)),
				# 	RNN(nn.Sequential(
				# 		nn.Linear(128, 128), nn.ReLU(),
				# 		nn.Linear(128, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True),
				# 		nn.Linear(64, 64), nn.ReLU(inplace=True)
				# 	)),
				# ]
				nn.LSTM(64, 64, batch_first=False, num_layers=1, bias=True, bidirectional=False),
			),
			[ # 输出每个动作对应的 Q 值
				nn.Sequential( # y/n/q
					nn.Linear(64, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, n_actions_ynq)
				),
				[ # inventory & enter
					( # inventory
						nn.Sequential( # 物品栏无序。这样可以消除输入的顺序性，生成每个物品的表示
							nn.Linear(6, 16), nn.Tanh(), # 由于物品栏的变化通常不大，不使用 batchnorm
							nn.Linear(16, 16), nn.ReLU(inplace=True),
							nn.Linear(16, 8), nn.ReLU(inplace=True),
							nn.Linear(8, 8), nn.ReLU(inplace=True),
							nn.Linear(8, 8), nn.ReLU(inplace=True),
						),
						nn.Sequential( # 联系角色状态、周围和本层地图，生成当前状态下每个物品的表示
							nn.Linear(64+8, 16), nn.Tanh(),
							nn.Linear(16, 16), nn.ReLU(inplace=True),
							nn.Linear(16, 16), nn.ReLU(inplace=True),
							nn.Linear(16, 4), nn.ReLU(inplace=True),
							nn.Linear(4, 4), nn.ReLU(inplace=True),
							nn.Linear(4, 4), nn.ReLU(inplace=True),
							nn.Linear(4, 1)
						),
						nn.Sequential( # 考虑多个物品间的联系。物品栏无序故不需检测连续介值的特征
							nn.Linear(55, 55), nn.Softmax(-1),
							nn.Linear(55, 55), nn.ReLU(inplace=True),
							nn.Linear(55, 55), nn.ReLU(inplace=True),
							nn.Linear(55, 55), nn.ReLU(inplace=True),
							nn.Linear(55, 55), nn.ReLU(inplace=True),
							nn.Linear(55, 55)
						),
					),
					nn.Sequential(
						nn.Linear(64, 8), nn.ReLU(inplace=True),
						nn.Linear(8, 8), nn.ReLU(inplace=True),
						nn.Linear(8, 1)
					),
				],
				nn.Sequential( # usual actions
					nn.Linear(64, 16), nn.Tanh(),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, n_actions_normal)
				),
			],
		]
		def add_q_to_model(self:nn.Module, q, name:str):
			if isinstance(q, nn.Module):
				self.add_module(name, q)
			else:
				for no, hierarchy in enumerate(q):
					add_q_to_model(self, hierarchy, name+'_'+str(no))
		add_q_to_model(self, self.Q, 'Q')
		self.INITIAL_RNN_STATE = (torch.zeros([1, 64]).to(self.device),)*2
		self.to(self.device)

	def initial_RNN_state(self):
		return self.INITIAL_RNN_STATE

	from typing import List, Tuple
	def _forward_y_seq(self, obs_batch:List[nle.basic.obs.observation]):
		''' output: Q.y(obs) '''
		map_4, srdng5x5_4, blstat_26, inv_55_6, misc_6 = preprocess(obs_batch, self.device)

		a = self.Q[0][0][0](map_4)
		a:torch.Tensor = self.Q[0][0][1](a)
		b = self.Q[0][1][0](srdng5x5_4)
		b:torch.Tensor = self.Q[0][1][1](b)
		c:torch.Tensor = self.Q[0][2](blstat_26)

		a = a.flatten(1) # [batch_size, 1, 2, 8] -> [batch_size, 16]
		b = b.flatten(1) # [batch_size, 9]

		y:torch.Tensor = torch.cat([a, b, c], axis=-1) # 合并
		# if self.training and len(y)==1: self.Q[1].train(False)
		y = self.Q[1][0](y) # batch size * 64
		# if self.training: self.Q[1].train(True)
		return y, (map_4, srdng5x5_4, blstat_26, inv_55_6, misc_6)
	def _forward_y(self, obs_batch:List[nle.basic.obs.observation], RNN_states:List[Tuple[torch.Tensor, torch.Tensor]]):
		assert len(obs_batch) == len(RNN_states)
		# RNN_states = torch.stack(RNN_states) # [batch_size][64]

		y, l = self._forward_y_seq(obs_batch)

		h:torch.Tensor = torch.stack(tuple(RNN_state[0] for RNN_state in RNN_states), axis=1)
		c:torch.Tensor = torch.stack(tuple(RNN_state[1] for RNN_state in RNN_states), axis=1)
		y = y.unsqueeze(0)
		y, (h, c) = self.Q[1][1](y, (h, c))
		y:torch.Tensor = y.squeeze(0)
		h, c = h.transpose(0, 1).detach(), c.transpose(0, 1).detach() # [batch_size][num_layers][num_channels]

		RNN_states = [*zip(h, c)] # [Tensor([64])]*batch_size
		return y, l, RNN_states

	def _forward_ynq_action(self, y_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_ynq)] '''
		z:torch.Tensor = self.Q[2][0](y_batch) # batch size * len(action_ynq)
		return z
	def _forward_inv_action(self, y_batch:torch.Tensor, inv_55_6_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_inv)] '''
		z = self.Q[2][1][0][0](inv_55_6_batch)
		tmp = torch.stack([y_batch]*z.shape[1], axis=-2)
		z = torch.cat([z, tmp], axis=-1)
		z = self.Q[2][1][0][1](z)
		z = z.squeeze(-1) # batch size * 55
		z:torch.Tensor = self.Q[2][1][0][2](z)
		tmp:torch.Tensor = self.Q[2][1][1](y_batch) # <CR>
		z = torch.cat([z, tmp], axis=-1)
		return z
	def _forward_normal_action(self, y_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_normal)] '''
		z:torch.Tensor = self.Q[2][2](y_batch) # batch size * len(actions_normal)
		return z
	def _forward_actions(self, y:torch.Tensor, inv_55_6:torch.Tensor, misc_6:List[List[int]]):
		no_action_set = [action_set_no(misc) for misc in misc_6]
		a, b, b_inv, c = [], [], [], []
		for misc_i, y_i, inv_i in zip(no_action_set, y, inv_55_6):
			if misc_i==0: l=a
			elif misc_i==1:
				l=b
				b_inv.append(inv_i)
			else: l=c # misc_i==2
			l.append(y_i)
		if len(a): a = torch.stack(a)
		if len(b): b, b_inv = (torch.stack(b), torch.stack(b_inv))
		if len(c): c = torch.stack(c)

		if len(a): a = self._forward_ynq_action(a)
		if len(b): b = self._forward_inv_action(b, b_inv)
		if len(c): c = self._forward_normal_action(c)

		y = [a, b, c]
		j = [0, 0, 0]
		Q = [torch.zeros(1)] * len(misc_6)
		for i, misc in enumerate(no_action_set):
			r = y[misc][j[misc]]
			j[misc] += 1
			Q[i] = r
		return Q
	def forward(self, obs_batch:List[nle.basic.obs.observation], RNN_states:List[Tuple[torch.Tensor, torch.Tensor]]):
		''' output: max_{action}(Q(obs)[action]) '''
		y, (_, _, _, inv_55_6, misc_6), RNN_states = self._forward_y(obs_batch, RNN_states)
		Q = self._forward_actions(y, inv_55_6, misc_6)
		return Q, RNN_states

	def save(self, filename:str):
		torch.save(self, filename)
	def load(self, filename:str):
		self.load_state_dict(torch.load(filename).state_dict())

# class DQN_RNN(DQN): # example
# 	def __init__(self, device, n_actions_ynq:int, n_actions_normal:int):
# 		super().__init__(device, n_actions_ynq, n_actions_normal)
# 		self.RNN = DQN_RNN.RNN_Module().to(device)
# 		self.INITIAL_RNN_STATE = self.RNN.INITIAL_RNN_STATE.to(device)
# 	def initial_RNN_state(self):
# 		return self.INITIAL_RNN_STATE
# 	def forward(self, *args, **kwargs):
# 		pass