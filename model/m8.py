if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
'''
在 m7 的基础上使用 DQN_RNN，即考虑中间状态。
'''

from model.DQN import action_set_no
from model.DQN_RNN import DQN_RNN, nle, torch

actions_ynq = [ord(ch) for ch in [
	'\x1b', 'y', 'n', '*',
]]

actions_inv = [0, ord('$'), ord('#')]+[*range(ord('a'), ord('z')+1)]+[*range(ord('A'), ord('Z')+1)]

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
	'\r', # enter
]]
from typing import List
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
		action_index:int = None # Q[i]=[0] if state[i] is None
		action:int = 255 # reset env
	else:
		from model.explore.glyphs import translate_messages_misc
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
			Q = Q0.data + Q1.data
			action_index = Q.argmax().item()
			if no_action_set == 1: # 物品栏；将位置映射到对应位置的字母
				actions = state.inv_letters # 
			else: # 其它情况；将位置映射到动作的 ord
				actions = actions_list[no_action_set]
			action = actions[action_index]
	return action, action_index

def optimize_batch(
	batch_action_index:List[int],
	batch_reward:List[float],
	loss_func,
	optimizer:torch.optim.Optimizer,
	Q_train:List[torch.Tensor],
	next_Q_train:List[torch.Tensor], next_Q_eval:List[torch.Tensor],
	gamma:float
):
	'''
	DQL optimize:
		Qfun(S, A) -> R + γ Qfun(S_, argmax<a>(Qfun_(S_, a)))
	in this case,
		Q+(S)[A] -> R + γ Q+(next_S)[argmax<a>(Q-(next_S)[a])]
	and Q+(S) = 0 if S is None
	'''
	p, y, r = [], [], []
	for i, (q_train, A, next_q_train, next_q_eval, R) in enumerate(zip(Q_train, batch_action_index, next_Q_train, next_Q_eval, batch_reward)):
		if q_train is not None:
			try:
				p.append(q_train[A])
				y.append(next_q_train[next_q_eval.argmax()].item() if next_q_train is not None else 0.) # 终止状态 Q 为 0
				r.append(R)
			except:
				print(i, A)
				print(q_train)
				raise
	if not len(p): return 0
	p = torch.stack(p) # prediction
	y = torch.tensor(y, device=p.device)
	y = torch.tensor(r, device=p.device) + gamma * y
	# print(p); print(y)

	loss:torch.Tensor = loss_func(p, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

def forward_batch(
	batch_state:List[nle.basic.obs.observation],
	RNN_STATE:List[List[torch.Tensor]],
	models:List[DQN_RNN]
):
	'''
	for index i, if batch_state[i] is None:
		Q{0,1}[i] = None;
	else:
		Q{0,1}[i] = model{0,1}.forward(batch_state[i])
	'''
	assert len(models)==len(RNN_STATE)
	RETURN_Q:List[List[torch.Tensor]] = [[None]*len(batch_state) for _ in models]
	RETURN_RNN_STATE = [[model.initial_RNN_state()]*len(batch_state) for model in models] # 终止状态的 RNN_STATE 清零
	non_final_mask = [s is not None for s in batch_state]
	if any(non_final_mask):
		INPUT_batch_state = [s for s in batch_state if s is not None] # non final state batch
		INPUT_RNN_STATE = [rnn_state for (rnn_state, s) in zip(RNN_STATE, batch_state) if s is not None]

		OUTPUT = [model.forward(INPUT_batch_state, rnn_state) for (model, rnn_state) in zip(models, INPUT_RNN_STATE)]
		# RETURN_Q[non_final_mask] = OUTPUT_Q
		j = 0
		for i, non_final in enumerate(non_final_mask):
			if non_final:
				for k, output in enumerate(OUTPUT):
					RETURN_Q[k][i] = output[0][j]
					RETURN_RNN_STATE[k][i] = output[1][j] # ['%016X'%(id(i[0])) for i in RETURN_RNN_STATE[0]]
				j += 1
	return RETURN_Q, RETURN_RNN_STATE

from nle_win.batch_nle import batch
def train_n_batch(
	start_epoch:int, num_epoch:int,
	env:batch,
	model0:DQN_RNN, model1:DQN_RNN,
	loss_func,
	optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer,
	last_batch_state:List[nle.basic.obs.observation],
	batch_action_index:List[int],
	batch_reward:List[float],
	batch_state:List[nle.basic.obs.observation],
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	RNN_STATE0:List[torch.Tensor], RNN_STATE1:List[torch.Tensor],
	last_RNN_STATE0:List[torch.Tensor], last_RNN_STATE1:List[torch.Tensor],
	gamma:float
): # ((None, None, None), (None, R1, S1), ..., (Ai, Ri, Si), (Aj, Rj, None), (None, Rk, Sk), ...)
	last_batch_state_buffer = [*(nle.basic.obs.observation*env.frcv.batch_size)()] # allocate memory
	def copy_state(dst:List[nle.basic.obs.observation], src:List[nle.basic.obs.observation], buffer:List[nle.basic.obs.observation]):
		from ctypes import memmove, sizeof, pointer
		for i, state in enumerate(src):
			if state is not None:
				dst[i] = buffer[i]
				memmove(pointer(dst[i]), pointer(state), sizeof(state))
			else:
				dst[i] = None

	from random import randint
	for n_ep in range(start_epoch, start_epoch+num_epoch):
		print('epoch %-6d'%(n_ep))

		(optimizer, model, RNN_STATE, next_Q_train, next_Q_eval) = (
			(optimizer0, model0, last_RNN_STATE0, Q0, Q1),
			(optimizer1, model1, last_RNN_STATE1, Q1, Q0),
		)[randint(0, 1)]
		[Q_train], _ = forward_batch(last_batch_state, [RNN_STATE], [model])
		loss = optimize_batch(batch_action_index, batch_reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)

		batch_action = [select_action(s, n_ep, q0, q1) for (s, q0, q1) in zip(batch_state, Q0, Q1)]
		batch_action_index, batch_action = [i[1] for i in batch_action], [i[0] for i in batch_action]

		copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
		observations = env.step(batch_action) # 注意 batch_state 的元素均指向 env.frcv 的成员的内存，step 会改变 batch_state
		# from nle_win.batch_nle import EXEC
		# EXEC('env.render(0)')
		batch_state = [i.obs if not i.done else None for i in observations] # 更新 None 状态
		batch_reward = [i.reward for i in observations] # 没有什么必要

		[last_RNN_STATE0, last_RNN_STATE1] = [RNN_STATE0, RNN_STATE1]
		[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(batch_state, [RNN_STATE0, RNN_STATE1], [model0, model1]) # 累积 RNN STATE
		print('loss %10.4f | a '%(loss), bytes(batch_action))

	return last_batch_state, batch_action_index, batch_reward, batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1

def __main__(*, nums_epoch:List[int], batch_size:int, gamma:float, use_gpu:bool, model0_file:str=None, model1_file:str=None):
	'''
	隔 num_epoch 轮 optimize 更新一次 loss|score|env 记录和参数文件（缓存/日志？），以防宕机
	main 函数执行完后保存参数为文件名带总 epoch 数、时间戳、模型和环境参数的新文件
	'''
	from torch import nn
	device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
	model0:DQN_RNN = torch.load(model0_file) if model0_file is not None else DQN_RNN(device, len(actions_ynq), len(actions_normal))
	model1:DQN_RNN = torch.load(model1_file) if model1_file is not None else DQN_RNN(device, len(actions_ynq), len(actions_normal))
	optimizer0 = torch.optim.Adam(model0.parameters(), lr=0.01)
	optimizer1 = torch.optim.Adam(model1.parameters(), lr=0.01)
	loss_func = nn.SmoothL1Loss()

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(batch_size, 'character="Val-Hum-Fem-Law", savedir=None, penalty_step=-1/64')

	last_batch_state:List[nle.basic.obs.observation] = [None]*batch_size
	Q0:List[torch.Tensor] = [torch.zeros(1)]*batch_size
	Q1:List[torch.Tensor] = [torch.zeros(1)]*batch_size
	RNN_STATE0 = [model0.initial_RNN_state()]*batch_size
	RNN_STATE1 = [model1.initial_RNN_state()]*batch_size
	last_RNN_STATE0:List[torch.Tensor] = [None]*batch_size
	last_RNN_STATE1:List[torch.Tensor] = [None]*batch_size

	batch_action_index = [select_action(state, 0, Q0, Q1) for state in last_batch_state] # 这里的 batch_action_index 为临时变量
	batch_action_index = ([i[0] for i in batch_action_index], [i[1] for i in batch_action_index])
	batch_state, batch_action_index = env.step(batch_action_index[0]), batch_action_index[1] # 这里的 batch_state 为临时变量
	batch_state, batch_reward = [i.obs if not i.done else None for i in batch_state], [i.reward for i in batch_state]

	start_epoch = 0
	for num_epoch in nums_epoch:
		last_batch_state, batch_action_index, batch_reward, batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1 = train_n_batch( # one period
			start_epoch, num_epoch, env, model0, model1, loss_func, optimizer0, optimizer1,
			last_batch_state, batch_action_index, batch_reward, batch_state, # reset all env, do not use input q
			Q0, Q1, # Q[i] will not be visited if state[i] is None
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			gamma,
		)
		start_epoch += num_epoch
		# TODO: 保存，定期全量备份，合适的函数名

	disconnect()
	return model0, model1

def pretrain():
	raise Exception('TODO')

if __name__ == '__main__':
	batch_size = 8 # 128 会爆显存
	num_epoch = 4
	gamma = 1
	use_gpu = False
	# torch.autograd.set_detect_anomaly(True) # DEBUG
	models = __main__(nums_epoch=[num_epoch]*3, batch_size=batch_size, gamma=gamma, use_gpu=use_gpu)