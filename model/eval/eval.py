if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../..'))
	del os, sys

from model.DRQN import DRQN, torch
from model.train import forward_batch
from nle_win.batch_nle import batch
import nle_win as nle

from typing import List, Tuple

def select_action(
	state:nle.basic.obs.observation,
	Q0:torch.Tensor, Q1:torch.Tensor
):
	'''return action, action_index'''
	if state is None: # Q0 is None and Q1 is None if state is None:
		action_index:int = None # Q[i]=[0] if state[i] is None
		action:int = 255 # reset env
		Q_sum = 0.
	else:
		from model.misc import action_set_no, actions_list
		from model.glyphs import translate_messages_misc
		no_action_set = action_set_no(translate_messages_misc(state))

		Q = Q0.data + Q1.data
		action_index = Q.argmax().item()
		Q_sum = Q[action_index].item()
		if no_action_set == 1: # 物品栏；将位置映射到对应位置的字母
			actions = [*state.inv_letters]+[ord('\r')] # inv 55 + cr
		else: # 其它情况；将位置映射到动作的 ord
			actions = actions_list[no_action_set]
		action = actions[action_index]
	print('Q sum\t%+.6f'%(Q_sum))
	return action, action_index

def test(
	start_epoch:int, num_epoch:int,
	env:batch, batch_state:List[nle.basic.obs.observation],
	model0:DRQN, model1:DRQN,
	Q0:List[torch.Tensor], Q1:List[torch.Tensor],
	RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]], RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]],
):
	scores = [0]*0
	for n_ep in range(start_epoch, start_epoch+num_epoch):
		batch_action = [select_action(s, q0, q1) for (s, q0, q1) in zip(batch_state, Q0, Q1)]
		_, batch_action = [i[1] for i in batch_action], [i[0] for i in batch_action]

		# copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
		scores_record_last_batch_state = [(int(state.blstats[20]), int(state.blstats[9])) if state is not None else (0, 0) for state in batch_state]

		observations = env.step(batch_action) # 注意 batch_state 的元素均指向 env.frcv 的成员的内存，step 会改变 batch_state
		batch_state = [i.obs if not i.done else None for i in observations] # 更新 None 状态
		batch_reward = [i.reward for i in observations] # 没有什么必要

		[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(batch_state, [RNN_STATE0, RNN_STATE1], [model0, model1]) # 累积 RNN STATE
		# loss = 0.
		print('Action: %s'%(bytes(batch_action).replace(b'\xff', b'!').replace(b'\x1b', b'Q').replace(b'\x04', b'D').replace(b'\r', b'N').replace(b'\x00', b'0').decode()))
		print('Reward: {}'.format(['%5.2f'%(reward) for reward in batch_reward]))

		# losses.append(loss)
		for i, (record, state) in enumerate(zip(scores_record_last_batch_state, batch_state)):
			if state is None: # 当前为终止状态，则上一状态的 blstats[20], blstats[9] 分别为 T, total score
				scores += [n_ep, record[0], record[1]]
				print('game #%d end. Time: %d, score: %d'%(i, record[0], record[1]))
		from nle_win.batch_nle import EXEC
		EXEC('env.render(0)')
	return batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, scores, scores_record_last_batch_state

def __main__(
	nums_epoch:List[int], batch_size:int, env_param:str, use_gpu:bool,
	model0_parameter_file:str, model1_parameter_file:str,
):
	device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
	from model.misc import actions_ynq, actions_normal
	# model0 = DRQN(device, len(actions_ynq), len(actions_normal))
	# model1 = DRQN(device, len(actions_ynq), len(actions_normal))
	# model0.load(model0_parameter_file)
	# model1.load(model1_parameter_file)
	model0:DRQN = torch.load(model0_parameter_file)
	model1:DRQN = torch.load(model1_parameter_file)
	model0.requires_grad_(False)
	model1.requires_grad_(False)

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(batch_size, env_param)

	batch_state:List[nle.basic.obs.observation] = [None]*batch_size
	Q0:List[torch.Tensor] = [None]*batch_size
	Q1:List[torch.Tensor] = [None]*batch_size
	RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]] = [model0.initial_RNN_state()]*batch_size
	RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]] = [model1.initial_RNN_state()]*batch_size

	from matplotlib import pyplot as plt
	# plt.ion()
	Scores = [0]*0
	start_epoch = 0
	for num_epoch in nums_epoch:
		(batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, scores, cscores
		) = test(start_epoch, num_epoch, env, batch_state,
			model0, model1, Q0, Q1, RNN_STATE0, RNN_STATE1
		)
		Scores += scores
		print(cscores)
		# plt.clf()
		# plt.plot()
		# plt.show()
	disconnect()
	plt.plot(
		[Scores[(i*3)+1] for i in range(len(Scores//3))], # T
		[Scores[(i*3)+2] for i in range(len(Scores//3))], # score
		marker='.', markersize=4,
		linewidth=0, # scatter
	)

if __name__ == '__main__':
	use_gpu = False
	model0_parameter_file = 'D:\\words\\RL\\project\\nle_model\\model\\dat\\[2022-0519-012344]DRQN0.pt'
	model1_parameter_file = 'D:\\words\\RL\\project\\nle_model\\model\\dat\\[2022-0519-012344]DRQN1.pt'
#	from model import setting
	__main__(
		model0_parameter_file=model0_parameter_file,
		model1_parameter_file=model1_parameter_file,
		use_gpu=use_gpu,
		batch_size=1,
		env_param='character="Val-Hum-Fem-Law", savedir=None',
		nums_epoch=[256]*4,
	)