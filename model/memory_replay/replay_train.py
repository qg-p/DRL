if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/../..'))
	del os, sys

from model.memory_replay.files import format_time

from model.DRQN import DRQN
from model.main import torch, nle, translate_messages_misc, action_set_no, actions_list

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

from model.memory_replay.files import try_to_cover_file, save_parameter_tmpfile, logfilexz_save_loss, logfilexz_read_loss
def replay_train(*,
	filename_dataset_xz:str,
	filename_parameter0_out:str, filename_parameter1_out:str,
	log_file_xz:str,
	filename_parameter0_in:str=None, filename_parameter1_in:str=None,
	n_episodes:int, LR_list:list,
	gamma:float
):
	assert n_episodes <= len(LR_list)
	if (try_to_cover_file(log_file_xz)): return
	if (try_to_cover_file(filename_parameter0_out)): return
	if (try_to_cover_file(filename_parameter1_out)): return

	model0 = DRQN(torch.device('cpu'))
	model1 = DRQN(torch.device('cpu'))
	if filename_parameter0_in is not None: model0.load(filename_parameter0_in)
	if filename_parameter1_in is not None: model1.load(filename_parameter1_in)
	model0.requires_grad_(False)
	model1.requires_grad_(False)

	loss_func = torch.nn.SmoothL1Loss()

	from dataset.dataset import ARS_dataset_xz
	dataset = ARS_dataset_xz(filename_dataset_xz)
	dataset = dataset.readall()

	from random import randint

	losses = []
	tmpfile = [None, None]
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
		action_index = index_of_action(state, action)
		for n_ep in range(len(dataset)-1):
			# copy_state(last_batch_state, batch_state, last_batch_state_buffer) # 要在 step 前复制
			no = randint(0, 1)
			(optimizer, model, RNN_STATE, ) = (
				(optimizer0, model0, last_RNN_STATE0, ),
				(optimizer1, model1, last_RNN_STATE1, ),
			)[no]
			model.requires_grad_(True)
			Q_train, _ = forward(state, RNN_STATE, model)
			model.requires_grad_(False)

			line = dataset[n_ep]
			reward, action = line.reward, line.action
			action_index = index_of_action(state, action)
			state = line.state if dataset[n_ep+1].action != 255 else None # next state in actual

			last_RNN_STATE0, last_RNN_STATE1 = RNN_STATE0, RNN_STATE1
			Q0, RNN_STATE0 = forward(state, RNN_STATE0, model0)
			Q1, RNN_STATE1 = forward(state, RNN_STATE1, model1)
			(next_Q_train, next_Q_eval, ) = (
				(Q0, Q1, ),
				(Q1, Q0, ),
			)[no]
			loss = optimize(action_index, reward, loss_func, optimizer, Q_train, next_Q_train, next_Q_eval, gamma)
			if n_ep % 10 == 0:
				print('%6d | %10.4e'%(n_ep, loss))#, bytes([action]).replace(b'\xff', b'!').replace(b'\x1b', b'Q').replace(b'\x04', b'D').replace(b'\r', b'N').decode()))
			# from nle_win.batch_nle import EXEC
			# EXEC('env.render(0)')
			losses.append(loss)

		# save tmpfile
		import os
		try:
			datdir = os.path.dirname(__file__)+'/dat/'
		except:
			datdir = os.getcwd()+'/dat/'
		del os
		time = format_time()
		tmpfile_new = ' Episode %d Epoch %d Time [%s].pt'%(episode, n_ep, time)
		tmpfile_new = (datdir+'DRQN0'+tmpfile_new, datdir+'DRQN1'+tmpfile_new)
		del time
		try: # 写入失败则 tmpfile 不更新
			if save_parameter_tmpfile(model0, tmpfile_new[0], tmpfile[0], force_write=False, do_not_cover=True):
				tmpfile[0] = tmpfile_new[0]
		except: pass
		try:
			if save_parameter_tmpfile(model1, tmpfile_new[1], tmpfile[1], force_write=False, do_not_cover=True):
				tmpfile[1] = tmpfile_new[1]
		except: pass
		del tmpfile_new
	# save parameters file
	if not save_parameter_tmpfile(model0, filename_parameter0_out, tmpfile[0], force_write=True):
		print('Fail to save model0.')
	if not save_parameter_tmpfile(model1, filename_parameter1_out, tmpfile[1], force_write=True):
		print('Fail to save model1.')
	logfilexz_save_loss(log_file_xz, losses)
	losses = logfilexz_read_loss(log_file_xz)
	from matplotlib import pyplot as plt
	plt.plot(losses, marker='.', markersize=1, linewidth=0)
	plt.show()

if __name__ == '__main__':
	from model import setting
	import os
	time = 'test'#format_time()
	datdir = os.path.dirname(__file__)+'/dat/'
	filename_dataset_xz = datdir + '1-Val-Hum-Fem-Law.ARS.dat.xz'
	filename_parameter0_out = datdir + 'model0[{}].pt'.format(time)
	filename_parameter1_out = datdir + 'model1[{}].pt'.format(time)
	log_file_xz = datdir + 'loss[{}].log.xz'.format(time)
	del os, time
	LR_list = [0.01]*4+[0.5]*8+[0.01]*12
	n_episodes = len(LR_list)
	replay_train(
		filename_dataset_xz=filename_dataset_xz,
		filename_parameter0_out=filename_parameter0_out,
		filename_parameter1_out=filename_parameter1_out,
		log_file_xz=log_file_xz,
		n_episodes=n_episodes,
		LR_list=LR_list, # [.05, .02, .01, .01],
		gamma=setting.gamma,
	)