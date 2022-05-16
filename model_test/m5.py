if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
# DQN_LSTM 的 Double DQN 训练样板。

from model_test.DQN import action_set_no
from model_test.DQN_RNN import DQN_LSTM, nle, torch
from model_test.replay_memory import Transition

actions_ynq = [ord(ch) for ch in [
	'\x1b', 'y', 'n', '*',
]]

actions_inv = [[0, ord('$'), ord('#')]+[*range(ord('a'), ord('z')+1)]+[*range(ord('A'), ord('Z')+1)]]

actions_normal = [ord(ch) for ch in [
	'k', 'l', 'h','j', 'u', 'n', 'b', 'y', # compass actions
	'<', '>', 's',
#	'c', # close
	'\x04', # kick
	',', # pick up
	# 后面的行动后接一个 * 将 misc 由 [1, 0, 0] 转为 [0, 0, 1]。
	'a', 'e', 'r', 'q', # apply, eat, read, quaff
	't', # throw
	'W', 'A', # wear, take off
	'w', # wield
	'\x1b', # esc
]]
from typing import List, Tuple
actions_list:List[List[int]]=[actions_ynq, actions_inv, actions_normal]
# 不使用 actions_normal_allowed
def select_action(
	state:nle.basic.obs.observation, n_ep:int,
	Q0:torch.Tensor, Q1:torch.Tensor
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
		EPS_BASE = .1*0
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
			Q = Q0[0].data + Q1[0].data
			action_index = Q.argmax().item()
			if no_action_set == 1: # 物品栏；将位置映射到对应位置的字母
				actions = state.inv_letters # 
			else: # 其它情况；将位置映射到动作的 ord
				actions = actions_list[no_action_set]
			action = actions[action_index]
	return action, action_index

def train_batch(
	last_batch_state:List[nle.basic.obs.observation],
	last_batch_action_index:List[int],
	batch_reward:List[float],
	model0:DQN_LSTM, model1:DQN_LSTM,
	loss_func,
	optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer,
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	last_RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]], last_RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]],
	gamma:float
):
	'''
	DQL optimize:
		Qfun(S, A) -> R + γ Qfun(S_, argmax<a>(Qfun_(S_, a)))
	in this case,
		Q+(last_S)[A] -> R + γ Q+(S)[argmax<a>(Q-(S)[a])]
	and Q+(S) = 0 if S is None
	'''
	from random import randint
	no = randint(0, 1)
	(train_model, optimizer, train_last_RNN_STATE, Q_train, Q_eval) = (
		(model0, optimizer0, last_RNN_STATE0, Q0, Q1),
		(model1, optimizer1, last_RNN_STATE1, Q1, Q0),
	)[no]
	train_last_RNN_STATE = [(h.detach(), c.detach()) for (h, c) in train_last_RNN_STATE]
	last_Q_train, _ = forward_batch(last_batch_state, train_model, train_last_RNN_STATE)
	p, y, r = [], [], []
	for (last_Q, A, Q, Q_, R, S) in zip(last_Q_train, last_batch_action_index, Q_train, Q_eval, batch_reward, last_batch_state):
		if S is not None:
			try:
				p.append(last_Q[A])
				y.append(Q[Q_.argmax().item()])
				r.append(R)
			except:
				print(last_Q)
				print(A)
				print(Q)
				print(Q_.argmax().item())
				raise
	if not len(p): return
	p = torch.stack(p) # prediction
	y = torch.stack(y).detach()
	y = torch.tensor(r).to(model0.device) + gamma * y
	print('p[a]', p)
	# y = y.detach()
	# print(p); print(y)
	# print(no, id(train_last_RNN_STATE))

	loss:torch.Tensor = loss_func(p, y)
	optimizer0.zero_grad()
	optimizer1.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

def another_train_batch(
	last_batch_state:List[nle.basic.obs.observation],
	last_batch_action_index:List[int],
	batch_reward:List[float],
	last_Q0:List[torch.Tensor], last_Q1:List[torch.Tensor],
	loss_func,
	optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer,
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	gamma:float
):
	'''
	DQL optimize:
		Qfun(S, A) -> R + γ Qfun(S_, argmax<a>(Qfun_(S_, a)))
	in this case,
		Q+(last_S)[A] -> R + γ Q+(S)[argmax<a>(Q-(S)[a])]
	and Q+(S) = 0 if S is None
	'''
	from random import randint
	no = randint(0, 1)
	(last_Q_train, optimizer, Q_train, Q_eval) = (
		(last_Q0, optimizer0, Q0, Q1),
		(last_Q1, optimizer1, Q1, Q0),
	)[no]
	p, y, r = [], [], []
	for (last_Q, A, Q, Q_, R, S) in zip(last_Q_train, last_batch_action_index, Q_train, Q_eval, batch_reward, last_batch_state):
		if S is not None:
			p.append(last_Q[A])
			y.append(Q[Q_.argmax().item()])
			r.append(R)
	if not len(p): return
	p = torch.stack(p) # prediction
	y = torch.stack(y).detach()
	y = torch.tensor(r) + gamma * y
	print(p); print(y)
	# print(no, id(train_last_RNN_STATE))

	loss:torch.Tensor = loss_func(p, y)
	optimizer0.zero_grad()
	optimizer1.zero_grad()
	loss.backward(retain_graph=True)
	optimizer.step()

	return loss.item()

def forward_batch(
	batch_state:List[nle.basic.obs.observation],
	model:DQN_LSTM,	RNN_STATE:List[Tuple[torch.Tensor, torch.Tensor]]
):
	'''
	for index i, if batch_state[i] is None:
		reset RNN_STATE{0,1}[i];
		set Q{0,1}[i] to None; # Tensor([0.]);
	else:
		(Q{}[], RNN_STATE{}[])<{0,1},[i]> = model{}.forward(batch_state[], RNN_STATE{}[])<{0,1},[i]>
	'''
	RETURN_Q, RETURN_RNN_STATE = [None]*len(batch_state), [model.initial_LSTM_state()]*len(batch_state)
	non_final_mask = [s is not None for s in batch_state]
	if any(non_final_mask):
		INPUT_batch_state = [s for s in batch_state if s is not None] # non final state batch
		INPUT_RNN_STATE = [RNN_STATE for (RNN_STATE, s) in zip(RNN_STATE, batch_state) if s is not None]

		OUTPUT_Q, OUTPUT_RNN_STATE = model.forward(INPUT_batch_state, INPUT_RNN_STATE)
		# RETURN_Q[non_final_mask], RETURN_RNN_STATE[non_final_mask] = OUTPUT_Q, OUTPUT_RNN_STATE
		for i, non_final in enumerate(non_final_mask):
			j = 0
			if non_final:
				RETURN_Q[i], RETURN_RNN_STATE[i] = OUTPUT_Q[j], OUTPUT_RNN_STATE[j]
				j += 1
	return RETURN_Q, RETURN_RNN_STATE

from nle_win.batch_nle import batch
def train_n_batch(
	start_epoch:int, num_epoch:int,
	env:batch,
	model0:DQN_LSTM, model1:DQN_LSTM,
	loss_func,
	optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer,
	batch_state:List[nle.basic.obs.observation],
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]], RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]],
	batch_action_index:List[int],
	gamma:float
):
	from ctypes import memmove, sizeof, pointer
	last_batch_state_buffer = [*(nle.basic.obs.observation*env.frcv.batch_size)()] # allocate memory
	last_batch_state = [state for state in last_batch_state_buffer]

	for n_ep in range(start_epoch, start_epoch+num_epoch):
		print('epoch %-6d'%(n_ep))
		for i, state in enumerate(batch_state):
			if state is not None:
				last_batch_state[i] = last_batch_state_buffer[i]
				memmove(pointer(last_batch_state[i]), pointer(state), sizeof(state))
			else:
				last_batch_state[i] = None

		last_batch_action_index = batch_action_index
		batch_action = [select_action(s, n_ep, q0, q1) for (s, q0, q1) in zip(batch_state, Q0, Q1)]
		batch_action_index, batch_action = [i[1] for i in batch_action], [i[0] for i in batch_action]

		observations = env.step(batch_action)
		from nle_win.batch_nle import EXEC
		EXEC('env.render(0)')
		batch_state:List[nle.basic.obs.observation] = [i.obs if not i.done else None for i in observations]
		batch_reward = [i.reward for i in observations]

		last_RNN_STATE0, last_RNN_STATE1 = RNN_STATE0, RNN_STATE1
		last_Q0, last_Q1 = Q0, Q1
		Q0, RNN_STATE0 = forward_batch(batch_state, model0, RNN_STATE0)
		Q1, RNN_STATE1 = forward_batch(batch_state, model1, RNN_STATE1)
		print('q0[a]', torch.tensor([q[i].item() for (q, i) in zip(Q0, batch_action_index) if q is not None]))
		print('q1[a]', torch.tensor([q[i].item() for (q, i) in zip(Q1, batch_action_index) if q is not None]))
		# print('action:', bytes(batch_action))
		# print('Q0', Q0, 'RNN_STATE0')
		# print('Q1', Q1, 'RNN_STATE1')
		# print('RNNSTT0:', RNN_STATE0[0][0])
		# print('last RNNSTT0:', last_RNN_STATE0[0][0])

		try:
			train_batch(last_batch_state, last_batch_action_index, batch_reward, model0, model1, loss_func, optimizer0, optimizer1, Q0, Q1, last_RNN_STATE0, last_RNN_STATE1, gamma)
			#another_train_batch(last_batch_state, last_batch_action_index, batch_reward, last_Q0, last_Q1, loss_func, optimizer0, optimizer1, Q0, Q1, gamma)
		except:
			print(batch_action_index, bytes(batch_action))
			raise
	return batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1

def __main__(*, num_epoch:int, batch_size:int, gamma:float, use_gpu:bool, model0_file:str=None, model1_file:str=None):
	from torch import nn
	device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
	model0:DQN_LSTM = torch.load(model0_file) if model0_file is not None else DQN_LSTM(device, len(actions_ynq), len(actions_normal))
	model1:DQN_LSTM = torch.load(model1_file) if model1_file is not None else DQN_LSTM(device, len(actions_ynq), len(actions_normal))
	optimizer0 = torch.optim.SGD(model0.parameters(), lr=0.01)
	optimizer1 = torch.optim.SGD(model1.parameters(), lr=0.01)
	loss_func = nn.SmoothL1Loss()

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(batch_size, 'character="Val-Hum-Fem-Law", savedir=None, penalty_step=-0.01')
	
	batch_state:List[nle.basic.obs.observation] = [None]*batch_size
	Q0:List[torch.Tensor] = [None]*batch_size
	Q1:List[torch.Tensor] = [None]*batch_size
	RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]] = [model0.initial_LSTM_state()]*batch_size
	RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]] = [model1.initial_LSTM_state()]*batch_size
	batch_action_index = [0]*batch_size

	a=[i.detach() for i in model0.LSTM.parameters()]
	for i in range(1):
		batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1 = train_n_batch(
			0, num_epoch, env, model0, model1, loss_func, optimizer0, optimizer1,
			batch_state, # reset all env, do not use input q
			Q0, Q1, # Q[i] will not be visited if state[i] is None
			RNN_STATE0, RNN_STATE1, # will reset rnn_state[i] if state[i] is none
			batch_action_index,
			gamma,
		)
		# TODO: 保存，定期全量备份，合适的函数名
	b=[i.detach() for i in model0.LSTM.parameters()]
	for a, b in zip(a, b):
		print(a==b)

	disconnect()
	return model0, model1

def pretrain():
	raise Exception('TODO')

if __name__ == '__main__':
	batch_size = 2 # 128 会爆显存
	num_epoch = 32
	gamma = .995
	use_gpu = False#True
	# torch.autograd.set_detect_anomaly(True) # DEBUG
	models = __main__(num_epoch=num_epoch, batch_size=batch_size, gamma=gamma, use_gpu=use_gpu)