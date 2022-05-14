if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
'''
Double DQN
'''

import torch
from torch import nn

from model.DQN import *
def select_action(state:nle.basic.obs.observation, model0:DQN, model1:DQN, n_ep:int): # 产生 action
	# Return: action index in action_list[no]
	from explore.glyphs import translate_messages_misc
	no_action_set = action_set_no(translate_messages_misc(state))

	EPS_INCR = 2.
	EPS_BASE = .1*0
	EPS_DECAY = 100
	from math import exp
	epsilon = EPS_BASE + EPS_INCR * exp(-n_ep/EPS_DECAY)

	import random
	if random.random()<epsilon: # epsilon-greedy
		_use_human_input = False
		#_use_human_input = True
		if _use_human_input: action = select_action_human_input(no_action_set, state)
		else: action = select_action_rand_action(no_action_set)
	else: # model decision
		Q = model0.forward([state])[0] + model1.forward([state])[0]
		action = Q.argmax().item()
		if no_action_set==2 and actions_normal_allowed[action]==False:
			Qlist = Q.tolist()
			qmax = None
			for i, (allowed, q) in enumerate(zip(actions_normal_allowed, Qlist)):
				if allowed:
					if qmax is None or qmax<q:
						qmax = q
						action = i
		print('Q predict: %.6f'%(Q[action].item()))
	return action, no_action_set
from model.DQN import exec_action
from replay_memory import Transition
from typing import List
def train_batch(batch:List[Transition], model0:DQN, model1:DQN, loss_func, optimizer0:torch.optim.Optimizer, optimizer1:torch.optim.Optimizer, gamma:float, device:torch.device):
	from random import randint
	(train_model, eval_model, optimizer) = (model0, model1, optimizer0) if randint(0, 1) else (model1, model0, optimizer1)

	batch_state = [t.state for t in batch]
	batch_action = [t.action for t in batch]
	batch_reward = [t.reward for t in batch]
	batch_next_state = [t.next_state for t in batch]

	non_final_mask = torch.tensor([t is not None for t in batch_next_state])
	non_final_next_state = [t for t in batch_next_state if t is not None]

	predict = train_model.forward(batch_state)
	predict = torch.stack([q[a] for q, a in zip(predict, batch_action)]) # shape = [batch size,]

	y = torch.zeros(len(batch)).to(device)
	if len(non_final_next_state):
		t = train_model.forward(non_final_next_state)
		s = eval_model.forward(non_final_next_state)
		# t = torch.stack([q.max() if s is not None else torch.zeros(1) for q, s in zip(y, batch_next_state)]) # same as predict's shape
		t = torch.stack([q[a.argmax()] for (q, a) in zip(t, s)])
		y[non_final_mask] = t
	y = y*gamma + torch.tensor(batch_reward).to(device)
	eval_predict = torch.stack([q[a] for q, a in zip(eval_model.forward(batch_state), batch_action)])
	print('{}\n{}\n{}\n{}'.format(predict.detach(), eval_predict.detach(), y.detach(), torch.tensor(batch_reward)))

	optimizer.zero_grad()
	loss:torch.Tensor = loss_func(predict, y)
	loss.backward()
	optimizer.step()
	return loss.item()
def __main__(*, step_fun=None, use_gpu=False, num_epoch=517, gamma=.995, lr=.01, batch_size=16, memory_sample, memory_push, batch_start:int=0):
	nle.connect()
	device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
	model = (DQN(device), DQN(device))
	# print([_ for _ in policy_net.parameters()])

	loss_func = nn.MSELoss()
	optimizer = (torch.optim.Adam(model[0].parameters(), lr=lr), torch.optim.Adam(model[1].parameters(), lr=lr))

	done = True

	actions_normal_chmod(0)

	loss_list = ([], [])

	import time
	t = time.perf_counter()
	for n_ep in range(num_epoch):
		if done: # start of an episode
			from copy import deepcopy
			nle.Exec('env.reset()')
			obs = deepcopy(nle.getobs())
		nle.Exec('env.render()')

		action_index, no = select_action(obs, model[0], model[1], n_ep)
		last_obs = obs
		obs, reward, done = exec_action(action_index, no, obs)

		step_loss = train_batch(
			[Transition(last_obs, action_index, reward, None if done else obs)],
			model[0], model[1], loss_func, optimizer[0], optimizer[1], gamma, device
		)
		if step_fun is not None: step_fun(n_ep)
		memory_push(last_obs, action_index, reward, None if done else obs, step_loss)
		print('epoch %-6d step_loss:  %.8g'%(n_ep, step_loss))
		loss_list[0].append(step_loss)

		if n_ep < batch_start: # replay memory batch
			continue
		batch = memory_sample(batch_size)
		batch_loss = train_batch(batch, model[0], model[1], loss_func, optimizer[0], optimizer[1], gamma, device)
		print('epoch %-6d batch_loss: %.8g'%(n_ep, batch_loss))
		loss_list[1].append(batch_loss)
	t = time.perf_counter()-t
	try:
		print('device: {}, time: {} s'.format(torch.cuda.get_device_name(device),t))
	except:
		print('time: {} s', t)
	nle.disconnect()
	test_input = obs if obs is not None else last_obs
	test_output = (model[0].forward([test_input])[0] + model[1].forward([test_input])[0]).to('cpu')
	from explore.glyphs import translate_messages_misc
	test_output = [*zip(test_output.tolist(), actions_list[action_set_no(translate_messages_misc(test_input))])]
	print(test_output)
	return loss_list

def prof():
	use_gpu=True

	from replay_memory import replay_memory_windowed_HLR
	memory_param = (64, 1) # test
	memory = replay_memory_windowed_HLR(*memory_param)
	def memory_push(last_obs, action_index, reward, obs, step_loss):
		return memory.push(last_obs, action_index, reward, obs, step_loss)
	def memory_sample(batch_size:int):
		return memory.sample(batch_size)

	prof_activities = [torch.profiler.ProfilerActivity.CPU]
	if use_gpu: prof_activities += [torch.profiler.ProfilerActivity.CUDA]
	with torch.profiler.profiler.profile(activities=prof_activities, profile_memory=True, with_flops=True, with_modules=True) as prof:
		def prof_step_fun(n_ep:int):
			if n_ep in [16, 19]:
				prof.step()
		__main__(prof_step_fun=prof_step_fun, use_gpu=use_gpu, memory_sample=memory_sample, memory_push=memory_push, num_epoch=20)
		# prof.stop()
	print(prof.key_averages().table(sort_by='self_cuda_time_total'))

def main():
	use_gpu=True

	from replay_memory import replay_memory_windowed_HLR
	memory_param = (128, 4) # to use
	memory = replay_memory_windowed_HLR(*memory_param)
	def memory_push(last_obs, action_index, reward, obs, step_loss):
		return memory.push(last_obs, action_index, reward, obs, step_loss)
	def memory_sample(batch_size:int):
		return memory.sample(batch_size)

	batch_size=16
	batch_start=batch_size*4
	num_epoch=batch_start+50

	step_losses, batch_losses = __main__(use_gpu=use_gpu, memory_sample=memory_sample, memory_push=memory_push, num_epoch=num_epoch, batch_size=batch_size, lr=0.01, gamma=0.995, batch_start=batch_start)
	from matplotlib import pyplot as plt
	from math import log
	plt.title('log(loss): step | batch')
	plt.plot(range(0, num_epoch), [log(i) for i in step_losses])
	plt.plot(range(batch_start, num_epoch), [log(i) for i in batch_losses])
	plt.show()

if __name__=='__main__':
	main()