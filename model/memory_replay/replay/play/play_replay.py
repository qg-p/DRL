if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../../../..'))
	del os, sys
# 源于 model_test.m8.py
# baseline 模型
from model.DRQN import DRQN
from model.SAR_dataset import SAR_dataset
from model.memory_replay.dataset.files import format_time, logfilexz_load_int
from model.memory_replay.replay.replay_memory import replay_memory_WHLR
from typing import List, Tuple
import torch
import nle_win as nle
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

def select_action(state:nle.basic.obs.observation):
	'player input'
	if state is None:
		print('game end')
		no = None
		action = 255
		action_index = None
		return action, action_index
	from getch import Getch
	from model.misc import actions_list, action_set_no as set_no
	from model.glyphs import translate_messages_misc as t
	last_no = no
	no = set_no(t(state))
	if no == 1:
		actionslist = [*state.inv_letters]+[ord('\r')]
		actions = [c for c in actionslist if c != 0]
	else:
		actionslist = actions_list[no]
		actions = actionslist
	# print available actions (keys)
	if last_no != no or no == 1:
		special_actions = {0:'<0>', 0x1b:'<Esc>', ord('\r'):'<CR>', 4:'<Ctrl-D>',}
		actions_print = ''
		for action in actions:
			if action in special_actions.keys():
				action = special_actions[action]
			else:
				action = chr(action)
			actions_print += action + ' '
		print(actions_print)
	# input action
	while True:
		action = Getch()[0]
		if no == 1 and actions[0] == 0:
			action = 0
		if action not in actions:
			print('Invalid action.')
		else: break
	action_index = actionslist.index(action)
	return action, action_index

from model.memory_replay.replay.train import optimize_batch, forward_batch, train_memory_batch

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
	replay_memory:replay_memory_WHLR, batch_replay_memory_transition_q:List[replay_memory_WHLR.Transition_q],
	replay_batch_size:int,
): # ((None, None, None), (None, R1, S1), ..., (Ai, Ri, Si), (Aj, Rj, None), (None, Rk, Sk), ...)
	assert all((
		env.frcv.batch_size == 1,
		len(batch_state) == env.frcv.batch_size,
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
		batch_action = [select_action() for _ in range(env.frcv.batch_size)]
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
		# 修饰 batch_reward
		batch_reward = [reward + penalty_death if state is None # 游戏结束（基本上等于死亡），给予较大的负激励
			else ( # 如果 T 没有变化（例如放下不存在的物品），环境不发生改变，且饥饿度不增加，判断为行动不立即生效，给予略微的负激励，防止游戏状态陷入死循环导致收敛到奇怪的地方
				0 if state.blstats[20]!=last_scores_record[0] else penalty_still_T
			) + ( # 如果行动非法（暂时只实现 0 输入（inv 选择 0）），给予较大的负激励
				0 if action != 0 else penalty_invalid_action
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

def __main__(*,
	nums_epoch:List[int], gamma:float, env_param:str,
	penalty_still_T:float, penalty_invalid_action:float, penalty_death:float,
	logfile:str,
	model0_file_out:str, model1_file_out:str,
	model0_file_in:str=None, model1_file_in:str=None,
	loss_logfile_xz:str=None, score_logfile_xz:str=None, replay_loss_logfile_xz:str=None,
	Learning_rate:float,
	replay_memory:replay_memory_WHLR, replay_batch_size:int,
):
	'''
	隔 num_epoch 轮 optimize 更新一次 loss|score|env 记录和参数文件（缓存/日志？），以防宕机。
	保存带总 epoch 数、时间戳、模型和环境参数的日志文件。
	'''
	from model.memory_replay.dataset.files import try_to_create_file, logfilexz_save_float, iter_tmpfile, format_time, logfilexz_load_float, logfilexz_save_int
	assert try_to_create_file(logfile) and try_to_create_file(logfile+'.xz')
	if loss_logfile_xz is not None: assert try_to_create_file(loss_logfile_xz) # 格式：double (raw)
	if replay_loss_logfile_xz is not None: assert try_to_create_file(replay_loss_logfile_xz) # 格式：double (raw)
	if score_logfile_xz is not None: assert try_to_create_file(score_logfile_xz) # 格式：int, int, int (raw)，#_epoch, blstats_T, blstats_score
	assert try_to_create_file(model0_file_out)
	assert try_to_create_file(model1_file_out)
	tmpfile = [None, None, None, None, None]

	losses = [0.]*0
	scores = [0]*0
	replay_losses = [0.]*0

	from torch import nn
	device = torch.device("cpu")
	from model.misc import actions_ynq, actions_normal
	model0:DRQN = DRQN(device, len(actions_ynq), len(actions_normal))
	model1:DRQN = DRQN(device, len(actions_ynq), len(actions_normal))
	if model0_file_in is not None: model0.load(model0_file_in)
	if model1_file_in is not None: model1.load(model1_file_in)
	optimizer0 = torch.optim.Adam(model0.parameters(), lr=Learning_rate)
	optimizer1 = torch.optim.Adam(model1.parameters(), lr=Learning_rate)
	loss_func = nn.SmoothL1Loss()

	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(1, env_param)

	# opening log
	filelog = open(logfile, 'ab+')
	filelog.writelines([(line+'\n').encode() for line in [
		'{} time'.format(format_time()),
		'parameters:',
		'\tnums_epoch={}'.format(nums_epoch),
		'\tbatch_size={}'.format(1),
		'\tgamma={}'.format(gamma),
		'\tenv_param={}'.format(env_param),
		'\tpenalty_still_T={}'.format(penalty_still_T),
		'\tpenalty_invalid_action={}'.format(penalty_invalid_action),
		'\tpenalty_death={}'.format(penalty_death),
		'\tlogfile={}'.format(logfile),
		'\tmodel0_file_out={}'.format(model0_file_out),
		'\tmodel1_file_out={}'.format(model1_file_out),
		'\tmodel0_file_in={}'.format(model0_file_in),
		'\tmodel1_file_in={}'.format(model1_file_in),
		'\tloss_logfile_xz={}'.format(loss_logfile_xz),
		'\treplay_loss_logfile_xz={}'.format(replay_loss_logfile_xz),
		'\tscore_logfile_xz={}'.format(score_logfile_xz),
		'\tLearning_rate={}'.format(Learning_rate),
		'\treplay_memory={}'.format(replay_memory),
		'\treplay_batch_size={}'.format(replay_batch_size),
		'Start.'
	]])
	filelog.close()
	del filelog

	batch_state:List[nle.basic.obs.observation] = [None]*(1)

	Q0:List[torch.Tensor] = [None]*(1)
	Q1:List[torch.Tensor] = [None]*(1)

	RNN_STATE0     :List[Tuple[torch.Tensor, torch.Tensor]] = [model0.initial_RNN_state()]*(1)
	RNN_STATE1     :List[Tuple[torch.Tensor, torch.Tensor]] = [model1.initial_RNN_state()]*(1)
	last_RNN_STATE0:List[Tuple[torch.Tensor, torch.Tensor]] = [None]*(1)
	last_RNN_STATE1:List[Tuple[torch.Tensor, torch.Tensor]] = [None]*(1)

	batch_replay_memory_transition_q:List[replay_memory_WHLR.Transition_q] = [None]*(1)

	start_epoch = 0
	# RNN_test = [None, None]
	# RNN_test[0] = model0.Q[1][1].forward(torch.ones([1, 1, 64]))[0].flatten(0)
	# test_state = replay_memory_dataset[0][0].state
	for num_epoch in nums_epoch:
		# before, _ = model0._forward_y_seq([test_state])
		print('Epoch %8d'%(start_epoch))
		(	batch_state, batch_replay_memory_transition_q, Q0, Q1,
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			losses_append, scores_append, replay_losses_append,
		) = train_n_batch( # one period
			start_epoch, num_epoch, env, model0, model1, loss_func, optimizer0, optimizer1,
			batch_state, # reset all env, do not use input q
			Q0, Q1, # Q[i] will not be visited if state[i] is None
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			gamma, penalty_still_T, penalty_invalid_action, penalty_death,
			replay_memory, batch_replay_memory_transition_q, replay_batch_size,
		)
		# after, _ = model0._forward_y_seq([test_state])
		# print((before!=after).any())
		# print(before)
		# print(after)
		from nle_win.batch_nle import EXEC
		try: EXEC('env.render(0)')
		except: pass
		# batch_state[0] = None # 测试：重置状态，模拟一个 episode 结束。测试结果：True
		# todo: 合适的函数名
		start_epoch += num_epoch
		losses += losses_append
		scores += scores_append
		replay_losses += replay_losses_append
		del losses_append, scores_append, replay_losses_append

		# save tmpfile, update logflie
		import os
		try:
			datdir = os.path.dirname(os.path.abspath(__file__))+'/../../../dat'
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
			datdir+tmpfile_new+'replay_loss.log.xz',
			datdir+tmpfile_new+'score.log.xz',
		)
		filelog.writelines([(line+'\n').encode() for line in [
			'{} time'.format(curtime),
			'Epoch {}'.format(start_epoch),
			'losses length {}'.format(len(losses)),
			'replay losses length {}'.format(len(replay_losses)),
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
			assert replay_loss_logfile_xz is not None and iter_tmpfile(tmpfile_new[3], tmpfile[3], force_write=False, do_not_cover=True)
			tmpfile[3] = tmpfile_new[3]
			logfilexz_save_float(tmpfile[3], replay_losses)
			filelog.write(('{} saved. (replay loss log)\n'.format(tmpfile[3])).encode())
		except: filelog.write(('{} skipped. (replay loss log)\n'.format(tmpfile_new[3])).encode())
		try:
			assert score_logfile_xz is not None and iter_tmpfile(tmpfile_new[4], tmpfile[4], force_write=False, do_not_cover=True)
			tmpfile[4] = tmpfile_new[4]
			logfilexz_save_int(tmpfile[4], scores)
			filelog.write(('{} saved. (score log)\n'.format(tmpfile[4])).encode())
		except: filelog.write(('{} skipped. (score log)\n'.format(tmpfile_new[4])).encode())
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
		'replay losses length {}'.format(len(replay_losses)),
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
	if replay_loss_logfile_xz is not None:
		try:
			assert iter_tmpfile(replay_loss_logfile_xz, tmpfile[3], force_write=True)
			logfilexz_save_float(replay_loss_logfile_xz, replay_losses)
			filelog.write(('{} saved successfully. (replay loss log)\n'.format(replay_loss_logfile_xz)).encode())
		except: filelog.write(('Error: Fail to Write replay loss logfile "{}". Last log file: {}. (replay loss log)\n'.format(replay_loss_logfile_xz, tmpfile[3])).encode())
	else: filelog.write(b'No replay loss log. (replay loss log)\n')
	if score_logfile_xz is not None:
		try:
			assert iter_tmpfile(score_logfile_xz, tmpfile[4], force_write=True)
			logfilexz_save_int(score_logfile_xz, scores)
			filelog.write(('{} saved successfully. (score log)\n'.format(score_logfile_xz)).encode())
		except: filelog.write(('Error: Fail to Write score logfile "{}". Last log file: {}. (score log)\n'.format(score_logfile_xz, tmpfile[4])).encode())
	else: filelog.write(b'No score log. (score log)\n')
	filelog.close()
	# compress logfile
	filelog = open(logfile, 'rb')
	filelog = filelog.read() + '{} saved successfully. (main log)'.format(logfile+'.xz').encode()
	iter_tmpfile(logfile+'.xz', logfile, force_write=True)
	from dataset.xzfile import xz_file
	logfilexz = xz_file(logfile+'.xz', WR_ONLY=True)
	logfilexz.append(filelog)
	logfilexz.close()

	if loss_logfile_xz is not None: # 验证是否能加载
		losses = logfilexz_load_float(loss_logfile_xz)
	if replay_loss_logfile_xz is not None:
		replay_losses = logfilexz_load_float(replay_loss_logfile_xz)
	if score_logfile_xz is not None:
		scores = logfilexz_load_int(score_logfile_xz)

	from matplotlib import pyplot as plt
	plt.figure()
	plt.title('Epoch - loss')
	plt.plot(losses, marker='.', markersize=2, linewidth=0)

	plt.figure()
	plt.title('Epoch - replay loss')
	plt.plot(replay_losses, marker='.', markersize=2, linewidth=0)

	if len(scores):
		scores_x = [scores[i*3+0] for i in range(len(scores)//3)]
		scores_T = [scores[i*3+1] for i in range(len(scores)//3)]
		scores_s = [scores[i*3+2] for i in range(len(scores)//3)]

		_, ax1 = plt.subplots()
		plt.title('Epoch - time, score')
		ax2 = ax1.twinx()
		lines = ax1.plot(scores_x, scores_T, label='T', c='C0') + ax2.plot(scores_x, scores_s, label='S', c='C1')
		plt.legend(lines, [l.get_label() for l in lines])

		plt.figure()
		plt.title('time - score')
		plt.plot(scores_T, scores_s, marker='.', markersize=4, linewidth=0)
	plt.show()
	# del filelog
	return model0, model1

if __name__ == '__main__':
	from model import setting
	num_epoch = 128
	nums_epoch = [num_epoch]*128
	model_file_tag = 'DRQN'
	replay_memory=replay_memory_WHLR(32, 4, 4)
	replay_batch_size=4

	model0_file_in = None
	model1_file_in = None

	import os
	try:
		datdir = os.path.dirname(os.path.abspath(__file__))+'/../../../dat'
	except:
		datdir = os.getcwd()+'/dat'
	datdir = os.path.normpath(datdir) + '\\' # datdir 为当前目录的 /dat/

	if os.path.isdir(datdir+'in'): # 最新（时间最大）的参数文件作为输入
		file_in_time = [ # dat/in 里每个 pt 文件的时间
			[
				(i, int(i.replace('-', '').strip('[]')))
				for i in i.split(model_file_tag) if '.pt' not in i
			][0] for i in os.listdir(datdir+'in\\') if model_file_tag in i
		] # [('[2022-0518-0010], 202205180010'), ...]
		file_in_time_max = [i[1] for i in file_in_time]
		file_in_time_max = file_in_time_max.index(max(file_in_time_max))
		file_in_time = file_in_time[file_in_time_max][0]

		model0_file_in = datdir+'in\\'+file_in_time+model_file_tag+'0.pt'
		model1_file_in = datdir+'in\\'+file_in_time+model_file_tag+'1.pt'
		del file_in_time_max, file_in_time
	del os
	print('model0_file_in = {}'.format(model0_file_in))
	print('model1_file_in = {}'.format(model1_file_in))

	from model.memory_replay.dataset.files import format_time
	curtime = format_time()
	model0_file_out = datdir+'[{}]{}0.pt'.format(curtime, model_file_tag)
	model1_file_out = datdir+'[{}]{}1.pt'.format(curtime, model_file_tag)
	loss_logfile_xz = datdir+'[{}]loss.xz'.format(curtime)
	replay_loss_logfile_xz = datdir+'[{}]replay_loss.xz'.format(curtime)
	score_logfile_xz = datdir+'[{}]score.xz'.format(curtime)
	logfile = datdir+'[{}]main.log'.format(curtime)
	del curtime, format_time

	models = __main__(
		nums_epoch=nums_epoch,
		gamma=setting.gamma, env_param='character="{}", savedir=None'.format(setting.character),
		penalty_still_T=setting.penalty_still_T,
		penalty_invalid_action=setting.penalty_invalid_action,
		penalty_death=setting.penalty_death,
		logfile=logfile,
		Learning_rate=setting.lr,
		model0_file_out=model0_file_out, model1_file_out=model1_file_out,
		loss_logfile_xz=loss_logfile_xz, score_logfile_xz=score_logfile_xz, replay_loss_logfile_xz=replay_loss_logfile_xz,
		model0_file_in=model0_file_in, model1_file_in=model1_file_in,
		replay_memory=replay_memory, replay_batch_size=replay_batch_size,
	)
