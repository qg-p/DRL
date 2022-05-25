if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../../..'))
	del os, sys
from model.DRQN import nle, torch, DRQN, action_set_no, translate_messages_misc, actions_list

from typing import List, Tuple, Callable
def select_action(
	state:nle.basic.obs.observation, Q0:torch.Tensor, Q1:torch.Tensor,
	epsilon:float
):
	'''
	Return: action, action_index
	epsilon: 每个 action 有 epsilon 概率被禁用，至少保留一个 action
	'''
	if state is None: # Q0 is None and Q1 is None if state is None:
		action_index:int = None # Q[i]=[0] if state[i] is None
		action:int = 255 # reset env
	else:
		no_action_set = action_set_no(translate_messages_misc(state))

		from random import randint
		Q = (Q0.data + Q1.data).to('cpu') # Q.shape = [len(Q)]
		rand_action_mask = torch.rand(Q.shape)>epsilon
		# rand_action_mask = rand_action_mask.to(Q.device)
		action_index = Q[rand_action_mask].argmax().item() if rand_action_mask.any() else randint(0, len(Q)-1)
		if no_action_set == 1: # 物品栏；将位置映射到对应位置的字母
			actions = [*state.inv_letters]+[ord('\r')] # 
		else: # 其它情况；将位置映射到动作的 ord
			actions = actions_list[no_action_set]
		action = actions[action_index]
		# print('Q sum\t%+.6f'%((Q0.data + Q1.data)[action_index].item()))
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
	return:
		total loss, [0 if q_train[i] is None else separated loss[i]]
	'''
	p, y, r = [], [], []
	nf_q_train_mask = [] # q_train[i] is not None
	for q_train, A, next_q_train, next_q_eval, R in zip(Q_train, batch_action_index, next_Q_train, next_Q_eval, batch_reward):
		nf_q_train_mask.append(q_train is not None)
		if q_train is not None:
			p.append(q_train[A])
			y.append(next_q_train[next_q_eval.argmax()].item() if next_q_train is not None else 0.) # 终止状态 Q 为 0
			r.append(R)
	sep_loss = torch.zeros(len(Q_train))
	if not len(p): return 0., [i.item() for i in sep_loss]
	p = torch.stack(p) # prediction
	y = torch.tensor(y, device=p.device)
	y = torch.tensor(r, device=p.device) + gamma * y
	# print('d %+.3f'%(y[0].item()-p[0].item()), end=' ')
	# print('p %+.3f'%(p[0].item()), end=' ')
	# print('r %+.3f'%(batch_reward[0]), end=' ')
	# print('a {}'.format(batch_action_index[0]))
	# print(p); print(y)

	loss:torch.Tensor = loss_func(p, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	nf_q_train_mask = torch.tensor(nf_q_train_mask)
	p = p.detach().to('cpu')
	y = y.to('cpu')
	sep_loss[nf_q_train_mask] = torch.stack([loss_func(p_, y_) for p_, y_ in zip(p, y)])
	return loss.item(), [i.item() for i in sep_loss]

def forward_batch(
	batch_state:List[nle.basic.obs.observation],
	RNN_STATE:List[List[Tuple[torch.Tensor, torch.Tensor]]],
	models:List[DRQN]
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
		INPUT_RNN_STATE = [[rnn_state for (rnn_state, s) in zip(RNN_state, batch_state) if s is not None] for RNN_state in RNN_STATE]

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

from model.memory_replay.replay.replay_memory import replay_memory_WHLR

def memory_push_sample(
	replay_memory:replay_memory_WHLR, replay_batch_size:int,
	last_batch_state:List[nle.basic.obs.observation], batch_action_index:List[int], batch_reward:List[float], batch_state:List[nle.basic.obs.observation],
	sep_loss:List[float], batch_replay_memory_transition_q:List[replay_memory_WHLR.Transition_q]
):#	0: call
	'''
	1: update memory
	2: sample memory
	3: return (hist batch state sequence, batch state, new batch transition q)
	'''
	batch_replay_memory_transition_q = [
		replay_memory.push(last_state, action_index, reward, next_state, loss, transition_q)
		for last_state, action_index, reward, next_state, loss, transition_q in zip(last_batch_state, batch_action_index, batch_reward, batch_state, sep_loss, batch_replay_memory_transition_q)
	]
	replay_sample = replay_memory.sample(replay_batch_size)
	memory_batch_hist_state = [[*state] for state in zip(*[sample.queue for sample in replay_sample])] # transpose into array: [len_queue][batch_size]
	memory_batch_action_index = [sample.data.action_index for sample in replay_sample]
	memory_batch_reward = [sample.data.reward for sample in replay_sample]
	memory_batch_state = [sample.data.state for sample in replay_sample]
	return replay_sample, (memory_batch_hist_state, memory_batch_action_index, memory_batch_reward, memory_batch_state), batch_replay_memory_transition_q

def train_memory_batch(
	replay_memory:replay_memory_WHLR, replay_batch_size:int,
	last_batch_state_copy:List[nle.basic.obs.observation], batch_action_index:List[int], batch_reward:List[float], batch_state_copy:List[nle.basic.obs.observation],
	sep_loss:List[float], batch_replay_memory_transition_q:List[replay_memory_WHLR.Transition_q],
	no:int, model0:DRQN, model1:DRQN, loss_func, optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer, gamma:float,
):
	# push memory, sample memory
	(	replay_sample,
		(memory_batch_hist_state, memory_batch_action_index, memory_batch_reward, memory_batch_state),
		batch_replay_memory_transition_q
	) = memory_push_sample(
		replay_memory, replay_batch_size,
		last_batch_state_copy, batch_action_index, batch_reward, batch_state_copy, sep_loss, batch_replay_memory_transition_q
	)
	# forward Q_train
	RNN_STATE0, RNN_STATE1 = [model0.initial_RNN_state()]*replay_batch_size, [model1.initial_RNN_state()]*replay_batch_size
	for memory_batch_last_state in memory_batch_hist_state:
		[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(memory_batch_last_state, [RNN_STATE0, RNN_STATE1], [model0, model1])
	Q_train = [Q0, Q1][no]
	# forward next_Q_train, next_Q_eval
	[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(memory_batch_state, [RNN_STATE0, RNN_STATE1], [model0, model1])
	del RNN_STATE0, RNN_STATE1
	Q0 = [Q.detach() if Q is not None else Q for Q in Q0] # 降低显存占用
	Q1 = [Q.detach() if Q is not None else Q for Q in Q1]
	(next_Q_train, next_Q_eval, optimizer) = (
		(Q0, Q1, optimizer0),
		(Q1, Q0, optimizer1),
	)[no]
	# backward
	replay_loss, replay_sep_loss = optimize_batch(memory_batch_action_index, memory_batch_reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
	# update memory loss
	for loss, transition_q in zip(replay_sep_loss, replay_sample):
		transition_q.step(loss)
	return replay_loss, batch_replay_memory_transition_q

from nle_win.batch_nle import batch
from model.SAR_dataset import SAR_dataset
def train_n_batch(
	start_epoch:int, num_epoch:int,
	env:batch,
	model0:DRQN, model1:DRQN,
	loss_func,
	optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer,
	batch_state:List[nle.basic.obs.observation],
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]], RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]],
	last_RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]], last_RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]],
	gamma:float, penalty_still_T:float, penalty_invalid_action:float, penalty_death:float,
	epsilon_function:Callable[[dict, dict], bool], epsilon_function_globals:dict,
	replay_dataset:List[SAR_dataset],
	replay_memory:replay_memory_WHLR, batch_replay_memory_transition_q:List[replay_memory_WHLR.Transition_q],
	replay_batch_size:int,
): # ((None, None, None), (None, R1, S1), ..., (Ai, Ri, Si), (Aj, Rj, None), (None, Rk, Sk), ...)
	assert all((
		len(batch_state) == env.frcv.batch_size + len(replay_dataset),
		len(Q0) == len(batch_state), len(Q0) == len(RNN_STATE0), len(RNN_STATE0) == len(last_RNN_STATE0),
		len(Q1) == len(batch_state), len(Q1) == len(RNN_STATE1), len(RNN_STATE1) == len(last_RNN_STATE1),
		len(batch_replay_memory_transition_q) == len(batch_state),
	))
	from random import randint
	from copy import deepcopy
	losses = [0.]*0
	scores = [0]*0
	replay_losses = [0.]*0
	# from model.dry_forward import dry_forward_batch

	last_batch_state_copy = [deepcopy(state) for state in batch_state] # last state 要在 step 前复制

	scores_record_last_batch_state = [(int(state.blstats[20]), int(state.blstats[9])) if state is not None else (0, 0) for state in batch_state]
	for n_ep in range(start_epoch, start_epoch+num_epoch):
		batch_action = [
			select_action(s, q0, q1,
				epsilon_function({
					'n_ep':n_ep,
					'game_no':game_no,
					'last_scores':scores_record_last_batch_state,
					'states':batch_state,
				}, epsilon_function_globals)
			)
			for game_no, s, q0, q1 in zip(range(env.frcv.batch_size), batch_state, Q0, Q1)
		] # 前 batch_size 个 env 的决策
		batch_action_index, batch_action = [i[1] for i in batch_action], [i[0] for i in batch_action]

		scores_record_last_batch_state = [(int(state.blstats[20]), int(state.blstats[9])) if state is not None else (0, 0) for state in batch_state]

		no = randint(0, 1)
		(optimizer, model, RNN_STATE, ) = (
			(optimizer0, model0, last_RNN_STATE0, ),
			(optimizer1, model1, last_RNN_STATE1, ),
		)[no]
		# model.requires_grad_(True)
		[Q_train], [RNN_STATE] = forward_batch(batch_state, [RNN_STATE], [model]) # 旧 state
		del RNN_STATE
		# model.requires_grad_(False)

		observations = env.step(batch_action) # 注意 batch_state 的元素均指向 env.frcv 的成员的内存，step 会改变 batch_state
		batch_state = [obs.obs if not obs.done else None for obs in observations] # 更新 None 状态
		batch_reward = [obs.reward for obs in observations]
		# 添加 replay memory 的数据
		observations = [dataset.next() for dataset in replay_dataset] # 来自 dataset 的数据
		batch_action += [obs.action for obs in observations]
		batch_action_index += [obs.action_index for obs in observations]
		batch_state += [obs.state for obs in observations]
		batch_reward += [obs.reward for obs in observations]
		# 修饰 batch_reward
		batch_reward = [reward + (
			penalty_death if state is None # 游戏结束（基本上等于死亡），给予较大的负激励
			else ( # 如果 T 没有变化（例如放下不存在的物品），环境不发生改变，且饥饿度不增加，判断为行动不立即生效，给予略微的负激励，防止游戏状态陷入死循环导致收敛到奇怪的地方
					0 if state.blstats[20]!=last_scores_record[0] else penalty_still_T
				) + ( # 如果行动非法（暂时只实现 0 输入（inv 选择 0）），给予较大的负激励
					0 if action != 0 else penalty_invalid_action
				)
			) for reward, state, last_scores_record, action in zip(batch_reward, batch_state, scores_record_last_batch_state, batch_action)
		]

		[last_RNN_STATE0, last_RNN_STATE1] = [RNN_STATE0, RNN_STATE1]
		[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(batch_state, [RNN_STATE0, RNN_STATE1], [model0, model1]) # 累积 RNN STATE
		Q0 = [Q.detach() if Q is not None else Q for Q in Q0] # 降低显存占用
		Q1 = [Q.detach() if Q is not None else Q for Q in Q1]
		(next_Q_train, next_Q_eval, ) = (
			(Q0, Q1, ),
			(Q1, Q0, ),
		)[no]
		loss, sep_loss = optimize_batch(batch_action_index, batch_reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
		# loss = 0.

		# memory replay using replay memory
		batch_state_copy = [deepcopy(state) for state in batch_state]
		replay_loss, batch_replay_memory_transition_q = train_memory_batch(
			replay_memory, replay_batch_size,
			last_batch_state_copy, batch_action_index, batch_reward, batch_state_copy, sep_loss, batch_replay_memory_transition_q,
			no, model0, model1, loss_func, optimizer0, optimizer1, gamma
		)
		last_batch_state_copy = batch_state_copy
		print('%8.2e %8.2e | %s'%(loss, replay_loss, bytes(batch_action).translate(bytes.maketrans(b'\xff\x1b\x04\r\x00', b'!QDN0')).decode()))

		losses.append(loss)
		replay_losses.append(replay_loss)
		for record, state in zip(scores_record_last_batch_state[:env.frcv.batch_size], batch_state[:env.frcv.batch_size]):
			if state is None: # 当前为终止状态，则上一状态的 blstats[20], blstats[9] 分别为 T, total score
				scores += [n_ep, record[0], record[1]]

	return batch_state, batch_replay_memory_transition_q, Q0, Q1, RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1, losses, scores, replay_losses
