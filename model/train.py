if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
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
	'''
	p, y, r = [], [], []
	for q_train, A, next_q_train, next_q_eval, R in zip(Q_train, batch_action_index, next_Q_train, next_Q_eval, batch_reward):
		if q_train is not None:
			p.append(q_train[A])
			y.append(next_q_train[next_q_eval.argmax()].item() if next_q_train is not None else 0.) # 终止状态 Q 为 0
			r.append(R)
	if not len(p): return 0.
	p = torch.stack(p) # prediction
	y = torch.tensor(y, device=p.device)
	y = torch.tensor(r, device=p.device) + gamma * y
	# print('predict\t%+.6f'%(p[0].item()))
	# print('label\t%+.6f'%(y[0].item()))
	# print('reward\t%+.6f'%(batch_reward[0]))
	# print(p); print(y)

	loss:torch.Tensor = loss_func(p, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

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
): # ((None, None, None), (None, R1, S1), ..., (Ai, Ri, Si), (Aj, Rj, None), (None, Rk, Sk), ...)
	assert all((
		len(batch_state) == env.frcv.batch_size + len(replay_dataset),
		len(Q0) == len(batch_state), len(Q0) == len(RNN_STATE0), len(RNN_STATE0) == len(last_RNN_STATE0),
		len(Q1) == len(batch_state), len(Q1) == len(RNN_STATE1), len(RNN_STATE1) == len(last_RNN_STATE1),
	))
	from random import randint
	losses = [0.]*0
	scores = [0]*0
	# from model.dry_forward import dry_forward_batch

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

		# copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
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
		loss = optimize_batch(batch_action_index, batch_reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
		# loss = 0.
		print('%10.4e | %s'%(loss, bytes(batch_action).translate(bytes.maketrans(b'\xff\x1b\x04\r\x00', b'!QDN0')).decode()))

		losses.append(loss)
		for record, state in zip(scores_record_last_batch_state[:env.frcv.batch_size], batch_state[:env.frcv.batch_size]):
			if state is None: # 当前为终止状态，则上一状态的 blstats[20], blstats[9] 分别为 T, total score
				scores += [n_ep, record[0], record[1]]

	return batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1, losses, scores
