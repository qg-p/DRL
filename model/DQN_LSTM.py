if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
import nle_win as nle

import torch
from torch import nn

from model.DQN import DQN
class DQN_LSTM(DQN):
	# 在 DQN 的基础上加一 LSTM。64 为 Q[2] 的输入通道数以及 Q[1][0] 的输入通道数，即中间变量 y 的通道数。
	# 该 LSTM 不改变 Q[1] Q[2] 间的输入输出通道数 (64)，嵌入位置为 Q[1][1]，串行。
	# 原 Q[1] 网络逻辑位置变为 Q[1][0]，实际位置不改动。
	def __init__(self, device, n_actions_ynq:int, n_actions_normal:int) -> None:
		super().__init__(device, n_actions_ynq, n_actions_normal)
		self.LSTM = nn.LSTM(64, 64, batch_first=False, num_layers=1, bias=True, bidirectional=False).to(device) # y
	# LSTM_states 为临时变量，坐标为 (epoch, episode)，相邻 epoch 同一 episode 的 LSTM_state 邻接。
	# LSTM_state 初始值为 0。
	def initial_LSTM_state(self):
		return (torch.zeros([1, 64]).to(self.device), torch.zeros([1, 64]).to(self.device))
	from typing import List, Tuple
	# List 算作一列，则 obs_batch 与 LSTM_states 行对齐。
	def _forward_y(self, obs_batch:List[nle.basic.obs.observation], LSTM_states:List[Tuple[torch.tensor, torch.tensor]]):
		'''
		LSTM_states:[
			(Tensor([num_layers=1, num_channels=64]), Tensor([num_layers=1, num_channels=64])),
			...
		]
		'''
		assert len(obs_batch) == len(LSTM_states)
		y, l = super()._forward_y(obs_batch)

		h:torch.Tensor = torch.stack(tuple(LSTM_state[0] for LSTM_state in LSTM_states), axis=1) # transpose, unsqueeze
		c:torch.Tensor = torch.stack(tuple(LSTM_state[1] for LSTM_state in LSTM_states), axis=1) # unsqueeze: num_layers == 1
		y = y.unsqueeze(0) # [1][batch_size][num_channels]. 1: 每个 env，一次 step 产生长度为 1 的 observation。

		y, (h, c) = self.LSTM.forward(y, (h, c)) # h.shape = c.shape = [num_layers][batch_size][num_channels]

		h, c = h.transpose(0, 1), c.transpose(0, 1) # [batch_size][num_layers][num_channels]
		y = y.squeeze(0) # [batch_size][num_channels]

		LSTM_states = [*zip(h, c)]
		return y, l, LSTM_states

	def forward(self, obs_batch:List[nle.basic.obs.observation], LSTM_states:List[Tuple[torch.tensor, torch.tensor]]):
		''' output: max_{action}(Q(obs)[action]), LSTM_state batch '''
		y, (_, _, _, inv_55_6, misc_6), LSTM_states = self._forward_y(obs_batch, LSTM_states)
		Q = self._forward_actions(y, inv_55_6, misc_6)
		return Q, LSTM_states