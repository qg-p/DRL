if __name__ == '__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys
'''
DQN_LSTM 的单 Q 学习版本，尽量简洁
'''
import nle_win as nle
import torch
from torch import nn

from model_test.DQN import action_set_no
from model_test.m5 import actions_list, actions_ynq, actions_inv, actions_normal

from model_test.DQN_RNN import DQN_LSTM

use_gpu = False
device = torch.device('gpu' if torch.cuda.is_available() and use_gpu else 'cpu')

# batch_size = 1
gamma = .995

model = DQN_LSTM(device, len(actions_ynq), len(actions_normal))

def select_action(
	state:nle.basic.obs.observation, n_ep:int,
	Q:torch.Tensor
):
	'''
	return action, action_index
	'''
	if state is None:
		action_index:int = 0 # Q[i]=[0] if state[i] is None
		action:int = 255 # reset env
	else:
		from model_test.explore.glyphs import translate_messages_misc
		no_action_set = action_set_no(translate_messages_misc(state))

		EPS_INCR = 2.
		EPS_BASE = 1+.1*0 # 暂时不使用 Q
		EPS_DECAY = 100
		from math import exp
		epsilon = EPS_BASE + EPS_INCR * exp(-n_ep/EPS_DECAY)

		import random
		if random.random()<epsilon: # epsilon-greedy
			if no_action_set == 1:
				inv_letters_55 = [*state.inv_letters]
				actions = [i for i in enumerate(inv_letters_55) if i[1]] # non-zero
				action_index = random.randint(0, len(actions)-1)
				action_index, action = actions[action_index]
			else:
				actions = actions_list[no_action_set]
				action_index = random.randint(0, len(actions)-1)
				action = actions[action_index]
		else:
			action_index = Q.argmax().item()
			if no_action_set == 1: # 物品栏；将位置映射到对应位置的字母
				actions = state.inv_letters # 
			else: # 其它情况；将位置映射到动作的 ord
				actions = actions_list[no_action_set]
			action = actions[action_index]
	return action, action_index
from typing import List, Tuple
def main():
	from nle_win.batch_nle import connect, disconnect, batch, EXEC
	loss_func = nn.L1Loss()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

	RNN_STATE = model.initial_LSTM_state()
	Q = None

	connect()
	env = batch(1, 'character="Val-Hum-Fem-Law", savedir=None, penalty_step=-0.01')
	from copy import deepcopy
	state = None

	start_LSTM_parameters = [*model.LSTM.parameters()]

	for n_ep in range(10):
		action, action_index = select_action(state, n_ep, None)

		last_state = deepcopy(state)
		observations = env.step([action])[0]
		EXEC('env.render(0)')
		state:nle.basic.obs.observation = observations.obs if not observations.done else None
		reward:float = float(observations.reward)

		RNN_STATE_detach = RNN_STATE[0].detach(), RNN_STATE[1].detach()
		last_Q = model.forward([last_state], [RNN_STATE])[0][0] if last_state is not None else None
		# last_Q = Q
		Q, RNN_STATE = model.forward([state], [model.initial_LSTM_state()])
		Q, RNN_STATE = Q[0], RNN_STATE[0]

		if last_Q is not None:
			p = last_Q[action_index]
			y = torch.tensor(reward)
			if Q is not None:
				y += gamma*Q.max().detach()
			print(p, y)
			loss = loss_func.forward(p, y)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('epoch %6d'%(n_ep), loss.item())

	end_LSTM_parameters = [*model.LSTM.parameters()]
	print([end==start for start, end in zip(start_LSTM_parameters, end_LSTM_parameters)])

	disconnect()

if __name__ == '__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys
	torch.autograd.set_detect_anomaly(True) # DEBUG
	main()