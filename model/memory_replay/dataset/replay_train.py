if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../../..'))
	del os, sys

from model.memory_replay.dataset.files import format_time

from model.DRQN import DRQN
from model.train import torch, nle, translate_messages_misc, action_set_no, actions_list

reverse_actions_list = [
	{value:index for index, value in enumerate(actions)}
	for actions in actions_list
]
def index_of_action(state:nle.basic.obs.observation, action:int):
	if state is None: return None # assert action = 255
	no = action_set_no(translate_messages_misc(state))
	if no == 1:
		index = 0
		for a in state.inv_letters:
			if a==action: break
			index += 1
		# else index = 56 (<CR>)
	else:
		index = reverse_actions_list[no][action]
	return index

def optimize(
	action_index:int, reward:float,
	loss_func,
	optimizer:torch.optim.Optimizer,
	Q_train:torch.Tensor, next_Q_train:torch.Tensor, next_Q_eval:torch.Tensor,
	gamma:float
):
	if Q_train is None: return 0.
	p = Q_train[action_index]
	y = next_Q_train[next_Q_eval.argmax()] if next_Q_train is not None else torch.zeros(1, device=Q_train.device)[0]
	y = torch.tensor(reward, device=p.device) + gamma * y

	# print(p), print(y)
	loss:torch.Tensor = loss_func(p, y)
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

	return loss.item()

def forward(state:nle.basic.obs.observation, RNN_STATE:torch.Tensor, model:DRQN):
	([RETURN_Q], [RETURN_RNN_STATE]) = ([None], [model.initial_RNN_state()]) if state is None else model.forward([state], [RNN_STATE])
	return RETURN_Q, RETURN_RNN_STATE

from model.memory_replay.dataset.files import try_to_create_file, iter_tmpfile, logfilexz_save_float, logfilexz_load_float
def replay_train(*,
	filename_dataset_xz:str,
	filename_parameter0_out:str, filename_parameter1_out:str,
	log_file_xz:str,
	filename_parameter0_in:str=None, filename_parameter1_in:str=None,
	n_episodes:int, LR_list:list,
	gamma:float
):
	assert n_episodes <= len(LR_list)
	assert try_to_create_file(log_file_xz)
	assert try_to_create_file(filename_parameter0_out)
	assert try_to_create_file(filename_parameter1_out)

	model0 = DRQN(torch.device('cpu'))
	model1 = DRQN(torch.device('cpu'))
	if filename_parameter0_in is not None: model0.load(filename_parameter0_in)
	if filename_parameter1_in is not None: model1.load(filename_parameter1_in)
	# model0.requires_grad_(False)
	# model1.requires_grad_(False)

	loss_func = torch.nn.SmoothL1Loss()

	from dataset.dataset import ARS_dataset_xz
	dataset = ARS_dataset_xz(filename_dataset_xz)
	dataset = dataset.readall()

	from random import randint

	losses = []
	tmpfile = [None, None, None]
	# optimized_test = [tuple(
	# 	[
	# 		(output[0][0].detach(), torch.stack(output[1][0]).detach())
	# 		for output in [model.forward([dataset[0].state], [model.initial_RNN_state()])]
	# 	][0]
	# 	for model in (model0, model1)
	# )]
	for episode in range(n_episodes):
		optimizer0 = torch.optim.Adam(model0.parameters(), lr=LR_list[episode])
		optimizer1 = torch.optim.Adam(model1.parameters(), lr=LR_list[episode])
		print('Episode', episode)
		state = None
		reward = None
		action = None # unknown
		RNN_STATE0 = model0.initial_RNN_state()
		RNN_STATE1 = model1.initial_RNN_state()
		last_RNN_STATE0 = None
		last_RNN_STATE1 = None
		last_T = 0
		action_index = index_of_action(state, action)
		losses_n = []
		for n_ep in range(len(dataset)-1):
			# copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
			no = randint(0, 1)
			(optimizer, model, RNN_STATE, ) = (
				(optimizer0, model0, last_RNN_STATE0, ),
				(optimizer1, model1, last_RNN_STATE1, ),
			)[no]
			# model.requires_grad_(True)
			Q_train, _ = forward(state, RNN_STATE, model)
			# model.requires_grad_(False)

			line = dataset[n_ep]
			reward, action = line.reward, line.action
			action_index = index_of_action(state, action)
			last_T = 0 if state is None else state.blstats[20]
			state = line.state if dataset[n_ep+1].action != 255 else None # next state in actual
			reward += setting.penalty_death if state is None else ( # 游戏结束
			# 如果 T 没有变化（例如放下不存在的物品），环境不发生改变，且饥饿度不增加，判断为行动不立即生效，给予略微的负激励，防止游戏状态陷入死循环导致收敛到奇怪的地方
				0 if state.blstats[20]!=last_T else setting.penalty_still_T
			) + ( # 如果行动非法（暂时只实现 0 输入（inv 选择 0）），给予较大的负激励
				0 if action != 0 else setting.penalty_invalid_action
			)

			last_RNN_STATE0, last_RNN_STATE1 = RNN_STATE0, RNN_STATE1
			Q0, RNN_STATE0 = forward(state, RNN_STATE0, model0)
			Q1, RNN_STATE1 = forward(state, RNN_STATE1, model1)
			(next_Q_train, next_Q_eval, ) = (
				(Q0, Q1, ),
				(Q1, Q0, ),
			)[no]
			loss = optimize(action_index, reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
			losses_n.append(loss)
			if n_ep % 50 == 49: #, bytes([action]).replace(b'\xff', b'!').replace(b'\x1b', b'Q').replace(b'\x04', b'D').replace(b'\r', b'N').decode()))
				print('%6d | %10.4e %10.4e %10.4e'%(n_ep+1, max(losses_n), min(losses_n), sum(losses_n)/len(losses_n)))
				losses += losses_n
				losses_n = []

		print('%6d | %10.4e %10.4e %10.4e'%(n_ep+1, max(losses_n), min(losses_n), sum(losses_n)/len(losses_n)))
		losses += losses_n
		losses_n = []

		# save tmpfile
		import os
		try:
			datdir = os.path.dirname(__file__)+'/dat'
		except:
			datdir = os.getcwd()+'/dat'
		datdir = os.path.normpath(datdir) + '\\'
		del os
		curtime = format_time()
		tmpfile_new = '[%s]Eps%d_Epc%d'%(curtime, episode, n_ep)
		tmpfile_new = (
			datdir+tmpfile_new+'DRQN0.pt',
			datdir+tmpfile_new+'DRQN1.pt',
			datdir+tmpfile_new+'loss.log.xz',
		)
		del curtime
		try: # 写入失败则 tmpfile 不更新
			assert iter_tmpfile(tmpfile_new[0], tmpfile[0], force_write=False, do_not_cover=True)
			tmpfile[0] = tmpfile_new[0]
			model0.save(tmpfile[0])
		except: pass
		try:
			assert iter_tmpfile(tmpfile_new[1], tmpfile[1], force_write=False, do_not_cover=True)
			tmpfile[1] = tmpfile_new[1]
			model1.save(tmpfile[1])
		except: pass
		try:# '>' -> 's'
			assert iter_tmpfile(tmpfile_new[2], tmpfile[2], force_write=False, do_not_cover=True)
			tmpfile[2] = tmpfile_new[2]
			logfilexz_save_float(tmpfile[2], losses)
		except: pass
		del tmpfile_new

	# optimized_test += [tuple(
	# 	[
	# 		(output[0][0].detach(), torch.stack(output[1][0]).detach())
	# 		for output in [model.forward([dataset[0].state], [model.initial_RNN_state()])]
	# 	][0]
	# 	for model in (model0, model1)
	# )]
	# print('model0 optimized:', (optimized_test[0][0][0]!=optimized_test[1][0][0]).any().item())
	# print('model1 optimized:', (optimized_test[0][1][0]!=optimized_test[1][1][0]).any().item())
	# print('model0 RNN optimized:', (optimized_test[0][0][1]!=optimized_test[1][0][1]).any().item())
	# print('model1 RNN optimized:', (optimized_test[0][1][1]!=optimized_test[1][1][1]).any().item())
	# save parameters file
	try:
		assert iter_tmpfile(filename_parameter0_out, tmpfile[0], force_write=True)
		model0.save(filename_parameter0_out)
	except: print('Fail to save model0.')
	try:
		assert iter_tmpfile(filename_parameter1_out, tmpfile[1], force_write=True)
		model1.save(filename_parameter1_out)
	except: print('Fail to save model1.')
	try:
		assert iter_tmpfile(log_file_xz, tmpfile[2], force_write=True)
		logfilexz_save_float(log_file_xz, losses)
		losses = logfilexz_load_float(log_file_xz)
		from matplotlib import pyplot as plt
		plt.plot(losses, marker='.', markersize=1, linewidth=0)
		plt.show()
	except: print('Fail to W/R loss logfile "{}"'.format(log_file_xz))

if __name__ == '__main__':
	from model import setting
	import os
	curtime = format_time() # 'test'
	datdir = os.path.dirname(__file__)+'/dat/'
	filename_dataset_xz = datdir + '1-Val-Hum-Fem-Law.ARS.dat.xz'
	filename_parameter0_out = datdir + '[{}]DRQN0.pt'.format(curtime)
	filename_parameter1_out = datdir + '[{}]DRQN1.pt'.format(curtime)
	log_file_xz = datdir + '[{}]loss.log.xz'.format(curtime)
	del os, curtime
	# LR_list = [0.01]+[0.5]+[0.1]*2+[0.01]#[0.01]*2+[0.5]*4+[0.1]*8+[0.01]*6
	LR_list = [0.1]*2
	n_episodes = len(LR_list)
	replay_train(
		filename_dataset_xz=filename_dataset_xz,
		filename_parameter0_out=filename_parameter0_out,
		filename_parameter1_out=filename_parameter1_out,
		log_file_xz=log_file_xz,
		n_episodes=n_episodes,
		LR_list=LR_list, # [.05, .02, .01, .01],
		gamma=setting.gamma,
		filename_parameter0_in='D:\\words\\RL\\project\\nle_model\\model\\dat\\in\\[2022-0519-181720]DRQN0.pt',
		filename_parameter1_in='D:\\words\\RL\\project\\nle_model\\model\\dat\\in\\[2022-0519-181720]DRQN1.pt',
	)