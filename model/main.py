if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
# 源于 model_test.m8.py
# baseline 模型

from model.DRQN import nle, torch, DRQN, action_set_no, translate_messages_misc, actions_list

from typing import List, Tuple
from model.memory_replay.files import format_time, logfilexz_load_int
def select_action(
	state:nle.basic.obs.observation, n_ep:int,
	Q0:torch.Tensor, Q1:torch.Tensor
):
	'''
	return action, action_index
	'''
	if state is None: # Q0 is None and Q1 is None if state is None:
		action_index:int = None # Q[i]=[0] if state[i] is None
		action:int = 255 # reset env
	else:
		no_action_set = action_set_no(translate_messages_misc(state))

		EPS_INCR = 2.
		EPS_BASE = .05
		EPS_DECAY = 500
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
				# if no_action_set == 2:
				# 	return ord('s'), actions_list[2].index(ord('s'))
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
	gamma:float
): # ((None, None, None), (None, R1, S1), ..., (Ai, Ri, Si), (Aj, Rj, None), (None, Rk, Sk), ...)
	from random import randint
	losses = [0.]*0
	scores = [0]*0
	for n_ep in range(start_epoch, start_epoch+num_epoch):
		# print('epoch %-6d'%(n_ep))

		batch_action = [select_action(s, n_ep, q0, q1) for (s, q0, q1) in zip(batch_state, Q0, Q1)]
		batch_action_index, batch_action = [i[1] for i in batch_action], [i[0] for i in batch_action]

		# copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
		scores_record_last_batch_state = [(int(state.blstats[20]), int(state.blstats[9])) if state is not None else (0, 0) for state in batch_state]
		no = randint(0, 1)
		(optimizer, model, RNN_STATE, ) = (
			(optimizer0, model0, last_RNN_STATE0, ),
			(optimizer1, model1, last_RNN_STATE1, ),
		)[no]
		# model.requires_grad_(True)
		[Q_train], _ = forward_batch(batch_state, [RNN_STATE], [model]) # 旧 state
		# model.requires_grad_(False)

		observations = env.step(batch_action) # 注意 batch_state 的元素均指向 env.frcv 的成员的内存，step 会改变 batch_state
		batch_state = [i.obs if not i.done else None for i in observations] # 更新 None 状态
		batch_reward = [i.reward for i in observations] # 没有什么必要

		[last_RNN_STATE0, last_RNN_STATE1] = [RNN_STATE0, RNN_STATE1]
		[Q0, Q1], [RNN_STATE0, RNN_STATE1] = forward_batch(batch_state, [RNN_STATE0, RNN_STATE1], [model0, model1]) # 累积 RNN STATE
		Q0 = [Q.detach() if Q is not None else Q for Q in Q0] # 降低显存占用
		Q1 = [Q.detach() if Q is not None else Q for Q in Q1]
		(next_Q_train, next_Q_eval, ) = (
			(Q0, Q1, ),
			(Q1, Q0, ),
		)[no]
		loss = optimize_batch(batch_action_index, batch_reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
		print('%10.4e | %s'%(loss, bytes(batch_action).replace(b'\xff', b'!').replace(b'\x1b', b'Q').replace(b'\x04', b'D').replace(b'\r', b'N').decode()))

		losses.append(loss)
		for record, state in zip(scores_record_last_batch_state, batch_state):
			if state is None: # 当前为终止状态，则上一状态的 blstats[20], blstats[9] 分别为 T, total score
				scores += [n_ep, record[0], record[1]]
		# from nle_win.batch_nle import EXEC
		# EXEC('env.render(0)')

	return batch_state, Q0, Q1, RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1, losses, scores

def __main__(*,
	nums_epoch:List[int], batch_size:int, gamma:float, env_param:str, use_gpu:bool, logfile:str,
	model0_file_out:str, model1_file_out:str,
	model0_file_in:str=None, model1_file_in:str=None,
	loss_logfile_xz:str=None, score_logfile_xz:str=None,
	Learning_rate:float
):
	'''
	隔 num_epoch 轮 optimize 更新一次 loss|score|env 记录和参数文件（缓存/日志？），以防宕机。
	保存带总 epoch 数、时间戳、模型和环境参数的日志文件。
	'''
	from model.memory_replay.files import try_to_create_file, logfilexz_save_float, iter_tmpfile, format_time, logfilexz_load_float, logfilexz_save_int
	assert try_to_create_file(logfile)
	if loss_logfile_xz is not None: assert try_to_create_file(loss_logfile_xz) # 格式：double (raw)
	if score_logfile_xz is not None: assert try_to_create_file(score_logfile_xz) # 格式：int, int, int (raw)，#_epoch, blstats_T, blstats_score
	assert try_to_create_file(model0_file_out)
	assert try_to_create_file(model1_file_out)
	tmpfile = [None, None, None, None]

	losses = [0.]*0
	scores = [0]*0

	from torch import nn
	device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
	from misc import actions_ynq, actions_normal
	model0:DRQN = torch.load(model0_file_in) if model0_file_in is not None else DRQN(device, len(actions_ynq), len(actions_normal))
	model1:DRQN = torch.load(model1_file_in) if model1_file_in is not None else DRQN(device, len(actions_ynq), len(actions_normal))
	# model0.requires_grad_(False)
	# model1.requires_grad_(False)
	optimizer0 = torch.optim.Adam(model0.parameters(), lr=Learning_rate)
	optimizer1 = torch.optim.Adam(model1.parameters(), lr=Learning_rate)
	loss_func = nn.SmoothL1Loss()

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(batch_size, env_param)

	# opening log
	filelog = open(logfile, 'ab+')
	filelog.writelines([(line+'\n').encode() for line in [
		'{} time'.format(format_time()),
		'parameters:',
		'\tnums_epoch={}'.format(nums_epoch),
		'\tbatch_size={}'.format(batch_size),
		'\tgamma={}'.format(gamma),
		'\tenv_param={}'.format(env_param),
		'\tuse_gpu={}'.format(use_gpu),
		'\tlogfile={}'.format(logfile),
		'\tmodel0_file_out={}'.format(model0_file_out),
		'\tmodel1_file_out={}'.format(model1_file_out),
		'\tmodel0_file_in={}'.format(model0_file_in),
		'\tmodel1_file_in={}'.format(model1_file_in),
		'\tloss_logfile_xz={}'.format(loss_logfile_xz),
		'\tscore_logfile_xz={}'.format(score_logfile_xz),
		'\tLearning_rate={}'.format(Learning_rate),
		'Start.'
	]])
	filelog.close()
	del filelog

	batch_state:List[nle.basic.obs.observation] = [None]*batch_size
	# batch_action_index:List[int]              = [None]*batch_size
	# batch_reward      :List[float]            = [None]*batch_size

	Q0:List[torch.Tensor] = [None]*batch_size
	Q1:List[torch.Tensor] = [None]*batch_size

	RNN_STATE0     :List[Tuple[torch.Tensor, torch.Tensor]] = [model0.initial_RNN_state()]*batch_size
	RNN_STATE1     :List[Tuple[torch.Tensor, torch.Tensor]] = [model1.initial_RNN_state()]*batch_size
	last_RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]] = [None]*batch_size
	last_RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]] = [None]*batch_size

	start_epoch = 0
	# RNN_test = [None, None]
	# RNN_test[0] = model0.Q[1][1].forward(torch.ones([1, 1, 64]))[0].flatten(0)
	for num_epoch in nums_epoch:
		(	batch_state, Q0, Q1,
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			losses_append, scores_append
		) = train_n_batch( # one period
			start_epoch, num_epoch, env, model0, model1, loss_func, optimizer0, optimizer1,
			batch_state, # reset all env, do not use input q
			Q0, Q1, # Q[i] will not be visited if state[i] is None
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			gamma,
		)
		# batch_state[0] = None # 测试：重置状态，模拟一个 episode 结束。测试结果：True
		# TODO: 合适的函数名
		start_epoch += num_epoch
		losses += losses_append
		scores += scores_append
		del losses_append, scores_append

		# save tmpfile, update logflie
		import os
		try:
			datdir = os.path.dirname(__file__)+'/dat'
		except:
			datdir = os.getcwd()+'/dat'
		datdir = os.path.normpath(datdir) + '\\' # DOS: '\\'
		del os
		filelog = open(logfile, 'ab+')
		curtime = format_time()
		tmpfile_new = '[%s]%d_'%(curtime, start_epoch)
		tmpfile_new = (
			datdir+tmpfile_new+'DRQN0.pt',
			datdir+tmpfile_new+'DRQN1.pt',
			datdir+tmpfile_new+'loss.log.xz',
			datdir+tmpfile_new+'score.log.xz',
		)
		filelog.writelines([(line+'\n').encode() for line in [
			'{} time'.format(curtime),
			'Epoch {}'.format(start_epoch),
			'losses length {}'.format(len(losses)),
			'scores length {}'.format(len(scores)),
		]])
		del curtime
		try: # 写入失败则 tmpfile 不更新
			assert iter_tmpfile(tmpfile_new[0], tmpfile[0], force_write=False, do_not_cover=True)
			tmpfile[0] = tmpfile_new[0]
			model0.save(tmpfile[0])
			filelog.write(('{} saved. (model0 parameter)\n'.format(tmpfile[0])).encode())
		except: filelog.write(('{} skipped. (model0 parameter)\n'.format(tmpfile_new[0])).encode())
		try:
			assert iter_tmpfile(tmpfile_new[1], tmpfile[1], force_write=False, do_not_cover=True)
			tmpfile[1] = tmpfile_new[1]
			model1.save(tmpfile[1])
			filelog.write(('{} saved. (model1 parameter)\n'.format(tmpfile[1])).encode())
		except: filelog.write(('{} skipped. (model1 parameter)\n'.format(tmpfile_new[1])).encode())
		try:
			assert loss_logfile_xz is not None and iter_tmpfile(tmpfile_new[2], tmpfile[2], force_write=False, do_not_cover=True)
			tmpfile[2] = tmpfile_new[2]
			logfilexz_save_float(tmpfile[2], losses)
			filelog.write(('{} saved. (loss log)\n'.format(tmpfile[2])).encode())
		except: filelog.write(('{} skipped. (loss log)\n'.format(tmpfile_new[2])).encode())
		try:
			assert score_logfile_xz is not None and iter_tmpfile(tmpfile_new[3], tmpfile[3], force_write=False, do_not_cover=True)
			tmpfile[3] = tmpfile_new[3]
			logfilexz_save_int(tmpfile[3], scores)
			filelog.write(('{} saved. (score log)\n'.format(tmpfile[3])).encode())
		except: filelog.write(('{} skipped. (score log)\n'.format(tmpfile_new[3])).encode())
		filelog.close()
		del tmpfile_new, filelog
	# RNN_test[1] = model0.Q[1][1].forward(torch.ones([1, 1, 64]))[0].flatten(0)
	# print((RNN_test[0]!=RNN_test[1]).any().item()) # 测试：RNN参数是否在变化。测试结果：True

	disconnect()
	
	filelog = open(logfile, 'ab+')
	filelog.writelines([(line+'\n').encode() for line in [
		'{} time'.format(format_time()),
		'End.',
		'Epoch {}'.format(start_epoch),
		'losses length {}'.format(len(losses)),
		'scores length {}'.format(len(scores)),
	]])
	# save parameters file
	try:
		assert iter_tmpfile(model0_file_out, tmpfile[0], force_write=True)
		model0.save(model0_file_out)
		filelog.write(('{} saved successfully. (model0 parameter)\n'.format(model0_file_out)).encode())
	except: filelog.write(('Error: Fail to save {}. Last parameter file: {}. (model0 parameter)\n'.format(model0_file_out, tmpfile[0])).encode())
	try:
		assert iter_tmpfile(model1_file_out, tmpfile[1], force_write=True)
		model1.save(model1_file_out)
		filelog.write(('{} saved successfully. (model1 parameter)\n'.format(model1_file_out)).encode())
	except: filelog.write(('Error: Fail to save {}. Last parameter file: {}. (model1 parameter)\n'.format(model0_file_out, tmpfile[1])).encode())
	if loss_logfile_xz is not None:
		try:
			assert iter_tmpfile(loss_logfile_xz, tmpfile[2], force_write=True)
			logfilexz_save_float(loss_logfile_xz, losses)
			filelog.write(('{} saved successfully. (loss log)\n'.format(loss_logfile_xz)).encode())
		except: filelog.write(('Error: Fail to Write loss logfile "{}". Last log file: {}. (loss log)\n'.format(loss_logfile_xz, tmpfile[2])).encode())
	else: filelog.write(b'No loss log. (loss log)\n')
	if score_logfile_xz is not None:
		try:
			assert iter_tmpfile(score_logfile_xz, tmpfile[3], force_write=True)
			logfilexz_save_int(score_logfile_xz, scores)
			filelog.write(('{} saved successfully. (score log)\n'.format(score_logfile_xz)).encode())
		except: filelog.write(('Error: Fail to Write score logfile "{}". Last log file: {}. (score log)\n'.format(score_logfile_xz, tmpfile[3])).encode())
	else: filelog.write(b'No score log. (score log)\n')
	filelog.close()

	if loss_logfile_xz is not None: # 验证是否能加载
		losses = logfilexz_load_float(loss_logfile_xz)
	if score_logfile_xz is not None:
		scores = logfilexz_load_int(score_logfile_xz)
	from matplotlib import pyplot as plt
	plt.figure()
	plt.title('Epoch - loss')
	plt.plot(losses, marker='.', markersize=1, linewidth=0)
	if len(scores):
		plt.figure()
		plt.title('Epoch - time span, total score of each terminated Episode')
		scores_x = [scores[i*3+0] for i in range(len(scores)//3)]
		scores_T = [scores[i*3+1] for i in range(len(scores)//3)]
		scores_s = [scores[i*3+2] for i in range(len(scores)//3)]
		plt.plot(scores_x, scores_T)
		plt.plot(scores_x, scores_s)
	plt.show()
	# del filelog
	return model0, model1

if __name__ == '__main__':
	from model import setting
	batch_size = 128
	num_epoch = 512
	use_gpu = True

	if use_gpu: torch.cuda.set_per_process_memory_fraction(1.)
	# torch.autograd.set_detect_anomaly(True) # DEBUG

	from model.memory_replay.files import format_time
	curtime = format_time()
	model0_file_out = 'model\dat\[{}]DRQN0.pt'.format(curtime)
	model1_file_out = 'model\dat\[{}]DRQN1.pt'.format(curtime)
	loss_logfile_xz = 'model\dat\[{}]loss.xz'.format(curtime)
	score_logfile_xz = 'model\dat\[{}]score.xz'.format(curtime)
	logfile = 'model\dat\[{}]main.log'.format(curtime)
	del curtime, format_time

	models = __main__(
		nums_epoch=[num_epoch]*64, batch_size=batch_size,
		gamma=setting.gamma, env_param='character="Val-Hum-Fem-Law", savedir=None, penalty_step={}'.format(setting.penalty_step),
		logfile=logfile,
		use_gpu=use_gpu, Learning_rate=setting.lr,
		model0_file_out=model0_file_out, model1_file_out=model1_file_out,
		loss_logfile_xz=loss_logfile_xz, score_logfile_xz=score_logfile_xz,
		model0_file_in='D:\\words\\RL\\project\\nle_model\\model\\memory_replay\\dat\\[2022-0517-225215]model0.pt',
		model1_file_in='D:\\words\\RL\\project\\nle_model\\model\\memory_replay\\dat\\[2022-0517-225215]model1.pt',
	)