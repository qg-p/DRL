if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys
# 源于 model_test.m8.py
# baseline 模型
from model.DRQN import DRQN
from model.train import train_n_batch
from model.memory_replay.files import format_time, logfilexz_load_int
from typing import List, Tuple
import torch
import nle_win as nle

def __main__(*,
	nums_epoch:List[int], batch_size:int, use_gpu:bool,
	gamma:float, epsilon_func:str, epsilon_func_locals:dict,
	env_param:str,
	penalty_still_T:float, penalty_invalid_action:float, penalty_death:float,
	logfile:str,
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

	from typing import Callable
	epsilon_lambda:Callable[[int, dict], float] = eval(epsilon_func)
	assert isinstance(epsilon_lambda, Callable) and epsilon_lambda.__code__.co_argcount==2

	losses = [0.]*0
	scores = [0]*0

	from torch import nn
	device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
	from misc import actions_ynq, actions_normal
	model0:DRQN = DRQN(device, len(actions_ynq), len(actions_normal))
	model1:DRQN = DRQN(device, len(actions_ynq), len(actions_normal))
	if model0_file_in is not None: model0.load(model0_file_in)
	if model1_file_in is not None: model1.load(model1_file_in)
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
		'\tpenalty_still_T={}'.format(penalty_still_T),
		'\tpenalty_invalid_action={}'.format(penalty_invalid_action),
		'\tpenalty_death={}'.format(penalty_death),
		'\tuse_gpu={}'.format(use_gpu),
		'\tlogfile={}'.format(logfile),
		'\tmodel0_file_out={}'.format(model0_file_out),
		'\tmodel1_file_out={}'.format(model1_file_out),
		'\tmodel0_file_in={}'.format(model0_file_in),
		'\tmodel1_file_in={}'.format(model1_file_in),
		'\tloss_logfile_xz={}'.format(loss_logfile_xz),
		'\tscore_logfile_xz={}'.format(score_logfile_xz),
		'\tLearning_rate={}'.format(Learning_rate),
		'\tepsilon_func={}'.format(epsilon_func),
		'\tepsilon_func_locals=\'\'\'{}\'\'\''.format(epsilon_func_locals),
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
		print('Epoch %8d'%(start_epoch))
		(	batch_state, Q0, Q1,
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			losses_append, scores_append
		) = train_n_batch( # one period
			start_epoch, num_epoch, env, model0, model1, loss_func, optimizer0, optimizer1,
			batch_state, # reset all env, do not use input q
			Q0, Q1, # Q[i] will not be visited if state[i] is None
			RNN_STATE0, RNN_STATE1, last_RNN_STATE0, last_RNN_STATE1,
			gamma, penalty_still_T, penalty_invalid_action, penalty_death,
			epsilon_lambda, epsilon_func_locals
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
			datdir = os.path.dirname(os.path.abspath(__file__))+'/dat'
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
	nums_epoch = [num_epoch]*64
	use_gpu = True
	model_file_tag = 'DRQN'

	if use_gpu: torch.cuda.set_per_process_memory_fraction(1.)
	# torch.autograd.set_detect_anomaly(True) # DEBUG

	epsilon_func='''
lambda n_ep, locals: [( # function
	lambda EPS_BASE, EPS_INCR_DECAY_LIST, n_ep, exp: [
		EPS_BASE + sum([
			EPS_INCR_DECAY[0] * exp(-n_ep/EPS_INCR_DECAY[1])
			for EPS_INCR_DECAY in EPS_INCR_DECAY_LIST
		])
	][0])( # constants
		0*.05, ( # .05
			(0*2, 500),
			(0*0.25, 5000),
		),
		n_ep, locals['exp'],
	), # function 1
	locals['RENDER'](
		n_ep, locals['get_hour'](locals['strftime']), locals['EXEC'],
	), # function 2
][0]
'''
	from math import exp
	from nle_win.batch_nle import EXEC
	from time import strftime
	print(EXEC)
	epsilon_func_locals = {
		'exp':exp,
		'strftime':strftime,
		'EXEC':EXEC,
		'RENDER':(lambda n_ep, hour, EXEC: [
			None if not ((hour>9 or hour<3) and n_ep) else
			EXEC('env.render({})'.format((n_ep//10) % (batch_size))),
			# print(EXEC),
		][0]),
		'get_hour': (lambda strftime: [
			int(strftime('%H'))
		][0]),
	}
	del exp, EXEC, strftime
	# print(eval(epsilon_func)(50000, epsilon_func_locals))
	# exit()

	model0_file_in = None
	model1_file_in = None

	import os
	try:
		datdir = os.path.dirname(os.path.abspath(__file__))+'/dat'
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

	from model.memory_replay.files import format_time
	curtime = format_time()
	model0_file_out = 'model\\dat\\[{}]{}0.pt'.format(curtime, model_file_tag)
	model1_file_out = 'model\\dat\\[{}]{}1.pt'.format(curtime, model_file_tag)
	loss_logfile_xz = 'model\\dat\\[{}]loss.xz'.format(curtime)
	score_logfile_xz = 'model\\dat\\[{}]score.xz'.format(curtime)
	logfile = 'model\\dat\\[{}]main.log'.format(curtime)
	del curtime, format_time

	models = __main__(
		nums_epoch=nums_epoch, batch_size=batch_size,
		gamma=setting.gamma, env_param='character="{}", savedir=None'.format(setting.character),
		penalty_still_T=setting.penalty_still_T,
		penalty_invalid_action=setting.penalty_invalid_action,
		penalty_death=setting.penalty_death,
		logfile=logfile,
		use_gpu=use_gpu, Learning_rate=setting.lr,
		model0_file_out=model0_file_out, model1_file_out=model1_file_out,
		loss_logfile_xz=loss_logfile_xz, score_logfile_xz=score_logfile_xz,
		model0_file_in=model0_file_in, model1_file_in=model1_file_in,
		epsilon_func=epsilon_func, epsilon_func_locals=epsilon_func_locals
	)