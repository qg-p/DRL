if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
import nle_win as nle
'''
搭建测试用模型
'''

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
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, groups=groups)
		self.fn = fn
		self.network = nn.Sequential(self.conv, self.fn)
		self.in_channels = in_channels
		self.out_channels = out_channels

	def forward(self, x):
		y = self.network(x)
		return y
class DQN(nn.Module):
	def __init__(self, device) -> None:
		super().__init__()
		self.device = device
		self.Q = [ # List 表示顺序，Tuple 表示并列
			(
				[ # seq, process map 4(channels) * 21 * 79
					nn.Sequential(
						nn.Conv2d(4, 64, 1), nn.Sigmoid(), # all grids are considered same
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
						nn.Conv2d(4, 8, 3, padding=1), nn.Sigmoid(),
						(Conv(8, 8, 3, nn.ReLU(inplace=True), 1) | Conv(8, 8, 3, nn.ReLU(inplace=True), 1)).init(),
						nn.Conv2d(8, 1, 3, padding=1), nn.Sigmoid(),
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
					nn.Linear(26, 26), nn.Sigmoid(), # abs. no bn.
					nn.Linear(26, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16),
				),
			),
			nn.Sequential(
				nn.Linear(21*79+9+16, 64), nn.ReLU(inplace=True),
				nn.Linear(64, 64), nn.ReLU(inplace=True),
				nn.Linear(64, 64), nn.ReLU(inplace=True),
				nn.Linear(64, 64), nn.ReLU(inplace=True),
				nn.Linear(64, 64), nn.ReLU(inplace=True),
				nn.Linear(64, 64), nn.ReLU(inplace=True)
			),
			[ # 输出每个动作对应的 Q 值
				nn.Sequential( # y/n/q
					nn.Linear(64, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, len(actions_ynq))
				),
				( # inventory
					nn.Sequential( # 物品栏无序。这样可以消除输入的顺序性，生成每个物品的表示
						nn.Linear(6, 16), nn.Sigmoid(), # 由于物品栏的变化通常不大，不使用 batchnorm
						nn.Linear(16, 16), nn.ReLU(inplace=True),
						nn.Linear(16, 8), nn.ReLU(inplace=True),
						nn.Linear(8, 8), nn.ReLU(inplace=True),
						nn.Linear(8, 8), nn.ReLU(inplace=True),
					),
					nn.Sequential( # 联系角色状态、周围和本层地图，生成当前状态下每个物品的表示
						nn.Linear(64+8, 16), nn.Sigmoid(),
						nn.Linear(16, 16), nn.ReLU(inplace=True),
						nn.Linear(16, 16), nn.ReLU(inplace=True),
						nn.Linear(16, 4), nn.ReLU(inplace=True),
						nn.Linear(4, 4), nn.ReLU(inplace=True),
						nn.Linear(4, 4), nn.ReLU(inplace=True),
						nn.Linear(4, 1)
					),
					nn.Sequential( # 考虑多个物品间的联系。物品栏无序故不需检测连续介值的特征
						nn.Linear(55, 55), nn.Softmax(55),
						nn.Linear(55, 55), nn.ReLU(inplace=True),
						nn.Linear(55, 55), nn.ReLU(inplace=True),
						nn.Linear(55, 55), nn.ReLU(inplace=True),
						nn.Linear(55, 55), nn.ReLU(inplace=True),
						nn.Linear(55, 55)
					),
				),
				nn.Sequential( # usual actions
					nn.Linear(64, 16), nn.Sigmoid(),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, 16), nn.ReLU(inplace=True),
					nn.Linear(16, len(actions_normal))
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
		self.to(self.device)

	from explore.glyphs import translate_glyph
	default_glyph = 2359 # ' '
	default_translation = translate_glyph(default_glyph)
	from typing import List
	def _forward_y(self, obs_batch:List[nle.basic.obs.observation]):
		''' output: Q.y(obs) '''
		from typing import List
		def preprocess(obs_batch:List[nle.basic.obs.observation], device:torch.device):
			from explore.glyphs import translate_glyphs
			batch_size = len(obs_batch)
			map_batch = [None] * batch_size
			surroundings_batch = [None] * batch_size
			blstats_batch = [None] * batch_size
			misc_batch = [None] * batch_size
			inv_batch = [None] * batch_size

			from explore.glyphs import translate_messages_misc, translate_inv
			for batch_i, obs in enumerate(obs_batch):
				misc_batch[batch_i] = translate_messages_misc(obs)
				inv_batch[batch_i] = translate_inv(obs)
				blstats_batch[batch_i] = obs.blstats
				map_4chnl = translate_glyphs(obs) # map in 4 channels
				map_batch[batch_i] = map_4chnl.tolist()
				srdng5x5_4chnl = [[[int(DQN.default_translation[0])]*5]*5, [[int(DQN.default_translation[1])]*5]*5, [[int(DQN.default_translation[2])]*5]*5, [[int(DQN.default_translation[3])]*5]*5]
				y, x = obs.tty_cursor; y -= 1
				for i in range(max(0, y-2), min(map_4chnl.shape[1], y+3)): # shape: (4, 21, 79,)
					for j in range(max(0, x-2), min(map_4chnl.shape[2], x+3)):
						srdng5x5_4chnl[0][i-(y-2)][j-(x-2)] = int(map_4chnl[0][i][j])
						srdng5x5_4chnl[1][i-(y-2)][j-(x-2)] = int(map_4chnl[1][i][j])
						srdng5x5_4chnl[2][i-(y-2)][j-(x-2)] = int(map_4chnl[2][i][j])
						srdng5x5_4chnl[3][i-(y-2)][j-(x-2)] = int(map_4chnl[3][i][j])
				surroundings_batch[batch_i] = srdng5x5_4chnl

			map_batch:torch.Tensor = torch.tensor(map_batch, dtype=torch.float, device=device)
			surroundings_batch:torch.Tensor = torch.tensor(surroundings_batch, dtype=torch.float, device=device)
			blstats_batch:torch.Tensor = torch.tensor(blstats_batch, dtype=torch.float, device=device)
			inv_batch:torch.Tensor = torch.tensor(inv_batch, dtype=torch.float, device=device)
			return map_batch, surroundings_batch, blstats_batch, inv_batch, misc_batch
		map_4, srdng5x5_4, blstat_26, inv_55_6, misc_6 = preprocess(obs_batch, self.device)

		a = self.Q[0][0][0](map_4)
		a:torch.Tensor = self.Q[0][0][1](a)
		b = self.Q[0][1][0](srdng5x5_4)
		b:torch.Tensor = self.Q[0][1][1](b)
		c:torch.Tensor = self.Q[0][2](blstat_26)

		a = a.flatten(1) # [batch_size, 1, 2, 8] -> [batch_size, 16]
		b = b.flatten(1) # [batch_size, 9]

		y:torch.Tensor = torch.cat([a, b, c], axis=1) # 合并
		# if self.training and len(y)==1: self.Q[1].train(False)
		y = self.Q[1](y) # batch size * 64
		# if self.training: self.Q[1].train(True)
		return y, (map_4, srdng5x5_4, blstat_26, inv_55_6, misc_6)

	def _forward_ynq_action(self, y_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_ynq)] '''
		# if self.training and len(y_batch)==1: self.Q[2][0].train(False)
		z:torch.Tensor = self.Q[2][0](y_batch) # batch size * len(action_ynq)
		# if self.training: self.Q[2][0].train(True)
		return z
	def _forward_inv_action(self, y_batch:torch.Tensor, inv_55_6_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_inv)] '''
		# if self.training and len(y_batch)==1:
		# 	self.Q[2][1][1].train(False)
		# 	self.Q[2][1][2].train(False)
		z = self.Q[2][1][0](inv_55_6_batch)
		tmp = torch.cat([y_batch]*z.shape[1], axis=1)
		z = torch.cat([z, tmp], axis=2)
		z = self.Q[2][1][1](z).squeeze(2) # batch size * 55
		z:torch.Tensor = self.Q[2][1][2](z)
		# if self.training:
		# 	self.Q[2][1][1].train(True)
		# 	self.Q[2][1][2].train(True)
		return z
	def _forward_normal_action(self, y_batch:torch.Tensor):
		''' output: Q(obs)[len(actions_normal)] '''
		# if self.training and len(y_batch)==1: self.Q[2][2].train(False)
		z:torch.Tensor = self.Q[2][2](y_batch) # batch size * len(actions_normal)
		# if self.training: self.Q[2][2].train(True)
		return z
	def forward(self, obs_batch:List[nle.basic.obs.observation]):
		''' output: max_{action}(Q(obs)[action]) '''
		y, (_, _, _, inv_55_6, misc_6) = self._forward_y(obs_batch)

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

		if len(a): a = self._forward_ynq_action(a) # no batchnorm
		if len(b): b = self._forward_inv_action(b, b_inv) # has batchnorm
		if len(c): c = self._forward_normal_action(c) # has batchnorm
				# c = self._forward_normal_action(torch.cat([c, c]))[0].unsqueeze(0) # has batchnorm

		y = [a, b, c]
		j = [0, 0, 0]
		Q = [torch.zeros(1)] * len(obs_batch)
		for i, misc in enumerate(no_action_set):
			r = y[misc][j[misc]]
			j[misc] += 1
			Q[i] = r
		return Q

# 三个 actions 序列，重要的是长度和 actions_normal 的内容
# 修改后需要到 exec_action 检查
actions_ynq=[ # q(uit), y(es), n(o)
	None, True, False
]
# actions_inv=[ # space, numbers, symbols and letters, return non-zero reward only-if misc[0] and not misc[1]
# 	chr(i) for i in range(33, 127)
# ] + [' ']
actions_inv=['-']+[chr(i) for i in range(ord('a'), ord('z'))]+[chr(i) for i in range(ord('A'), ord('Z'))]+['*', '?']
actions_normal=[ # normal actions
	0, 1, 2, 3, 4, 5, 6, 7, # move
	16, 17, 75, # up, down, search(wait)
#	19, # more
	30, # close (door)
#	38, # esc
#	39, # fight
	48, # kick
#	57, # open (door)
	61, # pick up
]
# 限定 actions_normal 中哪些动作可使用，不改变网络结构，作用于 select_action
# 以免收敛到全程 search 或只做错误动作，训练上百轮 T 还是 1。缓解稀疏奖励。
actions_normal_allowed=[True]*len(actions_normal)
def actions_normal_chmod(mode:int,
	modes:int=[
		[True]*8+[False]*(len(actions_normal)-8), # just movement*8
		[True]*8+[True]*3+[False]*(len(actions_normal)-8-3), # movement, up/dn, search
		[True]*len(actions_normal), # all
	]
):
	allowed = modes[mode] if mode>=0 and mode<len(modes) else []
	for i, mod in enumerate(allowed): # deep copy
		actions_normal_allowed[i] = mod
# actions = actions_ynq + actions_inv + actions_normal
actions_list = [actions_ynq, actions_inv, actions_normal]
def action_set_no(misc_6:list):
	if misc_6[0] or (misc_6[3] and not misc_6[4]): return 1
	elif misc_6[4] or misc_6[1]:
		return 0
	else: return 2

def select_action(state:nle.basic.obs.observation, model:DQN, n_ep:int): # 产生 action
	# Return: action index in action_list[no]
	from explore.glyphs import translate_messages_misc
	no_action_set = action_set_no(translate_messages_misc(state))

	EPS_INCR = 2.
	EPS_BASE = .1*0
	EPS_DECAY = 100
	from math import exp
	epsilon = EPS_BASE + EPS_INCR * exp(-n_ep/EPS_DECAY)

	import random
	if random.random()<epsilon: # epsilon-greedy
		def rand_action():
			actions = actions_list[no_action_set]
			if no_action_set!=2:
				action = random.randint(0, len(actions)-1)
			else:
				n_action = sum(actions_normal_allowed)
				n_action = random.randint(0, n_action-1)
				action = 0
				for i, allowed in enumerate(actions_normal_allowed):
					if allowed:
						if n_action:
							n_action -= 1
						else:
							action = i
							break
			return action
		def human_input(obs:nle.basic.obs.observation):
			from getch import Getch
			print('>>> ', end='')
			action = Getch().decode()
			print(action)
			try:
				if no_action_set==1:
					from ctypes import string_at, c_uint8
					inv_letter = (c_uint8*len(obs.inv_letters))()
					for i, letter in enumerate(obs.inv_letters): inv_letter[i] = letter
					inv = string_at(inv_letter)
					action = inv.index(action)
				else:
					if no_action_set==2:
						action = {
							'k': 0, 'l': 1, 'j': 2, 'h': 3, 'u': 4, 'n': 5, 'b': 6, 'y': 7,
							'<': 16, '>': 17, '.': 75, 's': 75,
							'c': 30, '\x04': 48, ',': 61, # close, kick, pick up
						}[action]
					elif no_action_set==0:
						action = {
							' ': None, '\x1b': None, 'y': True, 'n': False,
						}[action]
					actions:list = actions_list[no_action_set]
					action = actions.index(action)
			except:
				action = 0
			return action
		_use_human_input = False
		#_use_human_input = True
		if _use_human_input: action = human_input(state)
		else: action = rand_action()
	else: # model decision
		Q = model.forward([state])[0]
		action = Q.argmax().item()
		if no_action_set==2 and actions_normal_allowed[action]==False:
			Qlist = Q.tolist()
			qmax = None
			for i, (allowed, q) in enumerate(zip(actions_normal_allowed, Qlist)):
				if allowed:
					if qmax is None or qmax<q:
						qmax = q
						action = i
		print('Q predict: %.6f'%(Q[action].item()))
	return action, no_action_set
def exec_action(action_index:int, no:int, obs:nle.basic.obs.observation):
	primitive:str = None
	if no == 1:
		from ctypes import c_char, string_at
		bfr = (c_char*2)()
		bfr[0]=obs.inv_letters[action_index]
		bfr = string_at(bfr)
		primitive = bfr # character. e.g. letter, number, etc.
	elif no == 0:
		if   action_index == 0: action = 38 # <esc>
		elif action_index == 1: primitive = 'y' # Yes
		elif action_index == 2: primitive = 'n' # No
	else: # no==2
		actions = actions_list[no]
		action = actions[action_index]
	if primitive is None:
		obs, reward, done = nle.step(action)
		print('action {} {}\t reward {}'.format(no, action, '%.4g'%(reward)))
	else:
		obs, reward, done = nle.step_primitive(bytes([ord(primitive)]))
		print('action {} {}\t reward {}'.format(no, primitive, '%.4g'%(reward)))
	from copy import deepcopy
	return deepcopy(obs), reward, done
from replay_memory import Transition
from typing import List
def train_batch(batch:List[Transition], train_model:DQN, eval_model:DQN, loss_func, optimizer:torch.optim.Optimizer, gamma:float, device:torch.device):
	batch_state = [t.state for t in batch]
	batch_action = [t.action for t in batch]
	batch_reward = [t.reward for t in batch]
	batch_next_state = [t.next_state for t in batch]

	non_final_mask = torch.tensor([t is not None for t in batch_next_state])
	non_final_next_state = [t for t in batch_next_state if t is not None]

	predict = train_model.forward(batch_state)
	predict = torch.stack([q[a] for q, a in zip(predict, batch_action)]) # shape = [batch size,]

	y = torch.zeros(len(batch)).to(device)
	if len(non_final_next_state):
		t = eval_model.forward(non_final_next_state)
		# t = torch.stack([q.max() if s is not None else torch.zeros(1) for q, s in zip(y, batch_next_state)]) # same as predict's shape
		t = torch.stack([q.max() for q in t])
		y[non_final_mask] = t
	y = y*gamma + torch.tensor(batch_reward).to(device)
	eval_predict = torch.stack([q[a] for q, a in zip(eval_model.forward(batch_state), batch_action)])
	print('{}\n{}\n{}\n{}'.format(predict.detach(), eval_predict.detach(), y.detach(), torch.tensor(batch_reward)))

	optimizer.zero_grad()
	loss:torch.Tensor = loss_func(predict, y)
	loss.backward()
	optimizer.step()
	return loss.item()
def __main__(*, step_fun=None, use_gpu=False, num_epoch=517, gamma=.995, lr=.01, batch_size=16, memory_sample, memory_push, batch_start:int=0, T_sync:int=10):
	nle.connect()
	device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
	policy_net = DQN(device)
	target_net = DQN(device) # 提升训练稳定性，但为何不用 DDQN
	# print([_ for _ in policy_net.parameters()])

	loss_func = nn.MSELoss()
	optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)


	done = True
	policy_net = policy_net.train()
	target_net = target_net.eval()

	actions_normal_chmod(0)

	import time
	t = time.perf_counter()
	for n_ep in range(num_epoch):
		if done: # start of an episode
			from copy import deepcopy
			nle.Exec('env.reset()')
			obs = deepcopy(nle.getobs())
		nle.Exec('env.render()')

		if n_ep % T_sync == 0: # sync
			target_net.load_state_dict(policy_net.state_dict())

		action_index, no = select_action(obs, target_net, n_ep)
		last_obs = obs
		obs, reward, done = exec_action(action_index, no, obs)

		step_loss = train_batch(
			[Transition(last_obs, action_index, reward, None if done else obs)],
			policy_net, target_net, loss_func, optimizer, gamma, device
		)
		if step_fun is not None: step_fun(n_ep)
		memory_push(last_obs, action_index, reward, None if done else obs, step_loss)
		print('epoch %-6d step_loss:  %.8g'%(n_ep, step_loss))

		if n_ep < batch_start: # replay memory batch
			continue
		batch = memory_sample(batch_size)
		batch_loss = train_batch(batch, policy_net, target_net, loss_func, optimizer, gamma, device)
		print('epoch %-6d batch_loss: %.8g'%(n_ep, batch_loss))
	t = time.perf_counter()-t
	try:
		print('device: {}, time: {} s'.format(torch.cuda.get_device_name(device),t))
	except:
		print('time: {} s', t)
	nle.disconnect()
	test_input = obs if obs is not None else last_obs
	test_output = target_net.forward([test_input])[0].to('cpu')
	from explore.glyphs import translate_messages_misc
	test_output = [*zip(test_output.tolist(), actions_list[action_set_no(translate_messages_misc(test_input))])]
	print(test_output)

def prof():
	use_gpu=True
	
	from replay_memory import replay_memory_windowed_HLR
	memory_param = (128, 8) # to use
	memory_param = (64, 1) # test
	memory = replay_memory_windowed_HLR(*memory_param)
	def memory_push(last_obs, action_index, reward, obs, step_loss):
		return memory.push(last_obs, action_index, reward, obs, step_loss)
	def memory_sample(batch_size:int):
		return memory.sample(batch_size)

	prof_activities = [torch.profiler.ProfilerActivity.CPU]
	if use_gpu: prof_activities += [torch.profiler.ProfilerActivity.CUDA]
	with torch.profiler.profiler.profile(activities=prof_activities, profile_memory=True, with_flops=True, with_modules=True) as prof:
		def prof_step_fun(n_ep:int):
			if n_ep in [16, 19]:
				prof.step()
		__main__(prof_step_fun=prof_step_fun, use_gpu=use_gpu, memory_sample=memory_sample, memory_push=memory_push, num_epoch=20)
		# prof.stop()
	print(prof.key_averages().table(sort_by='self_cuda_time_total'))

def main():
	use_gpu=True
	
	from replay_memory import replay_memory_windowed_HLR
	memory_param = (128, 8) # to use
	memory = replay_memory_windowed_HLR(*memory_param)
	def memory_push(last_obs, action_index, reward, obs, step_loss):
		return memory.push(last_obs, action_index, reward, obs, step_loss)
	def memory_sample(batch_size:int):
		return memory.sample(batch_size)
	def step(n_ep, *args, **kwargs):
		import time
		if n_ep>40:
			time.sleep(10)

	__main__(step_fun=step, use_gpu=use_gpu, memory_sample=memory_sample, memory_push=memory_push, num_epoch=500, batch_size=4, lr=0.01, gamma=0.995, batch_start=4*8, T_sync=1)

if __name__=='__main__':
	main()