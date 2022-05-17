# run this file with python with nethack learning environment in linux (WSL)
from ctypes import POINTER, c_float, c_int32, sizeof, cast
from .basic import frame

class NLEnv():
	def __init__(self, env_id='NetHackChallenge-v0', *args, **kwargs) -> None:
		import nle
		from nle.env import NLE
		from .basic.common import observation, reward, done, info
		self.env:NLE = nle.env.gym.make(env_id, *args, **kwargs)
		self.observation = observation
		self.reward = reward
		self.done = done
		self.info = info
	def reset(self):
		self.observation = self.env.reset()
		return self.observation
	def step(self, action):
		self.observation, self.reward, self.done, self.info = self.env.step(action)
		return self.observation, self.reward, self.done, self.info
	def close(self):
		self.env.close()
	def render(self, mode='human'):
		return self.env.render(mode)
	def seed(self, seed:int):
		return self.env.seed(seed)
	def __del__(self):
		self.close()
	def primitive_step(self, action):
		# from: https://github.com/facebookresearch/nle/blob/f4f750af3cb07c095869c36a9bc1994edf438aa1/nle/env/base.py#L337
		# time: 2022/5/11 (latest)
		last_observation = tuple(a.copy() for a in self.env.last_observation)

		observation, done = self.env.env.step(action) # (obs tuple, reward)
		is_game_over = observation[self.env._program_state_index][0] == 1
		if is_game_over or not self.env._allow_all_modes:
			observation, done = self.env._perform_known_steps(
				observation, done, exceptions=True
			)

		self.env._steps += 1
		self.env.last_observation = observation

		if self.env._check_abort(observation):
			end_status = self.env.StepStatus.ABORTED
		else:
			end_status = self.env._is_episode_end(observation)
		end_status = self.env.StepStatus(done or end_status)

		self.reward = float(self.env._reward_fn(last_observation, None, observation, end_status)) # action is not used so it's safe

		if end_status and not done:
			self.env._quit_game(observation, done)
			done = True

		self.observation = self.env._get_observation(observation)
		self.done:bool = done
		return self.observation, self.reward, self.done

env = NLEnv(character='@', savedir=None, penalty_step=-0.01)

# return: whether sent/received successfully
def errno_check(f:frame, errNo:int)->bool:
	from .basic import Type, ErrNo
	if errNo in (ErrNo['success'], ErrNo['warn'], ErrNo['other']):
		return True
	elif errNo == ErrNo['too_long']:
		_ptr = (c_int32*1)(f.f.ptr_len)
		fsnd = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['errNo'])
		errNo = fsnd.send() # 仅 success、cannot_open、cannot_write
		return False
	elif errNo in (ErrNo['cannot_open'], ErrNo['cannot_read'], ErrNo['cannot_write']):
		return False
	else:
		from sys import _getframe
		print('%s(%d), %s: Error: unknown errNo' % (__file__, _getframe().f_lineno, errno_check.__name__))
		return True

def sendn(fsnd:frame, n:int=3):
	for _ in range(n):
		errNo = fsnd.send()
		check_pass = errno_check(fsnd, errNo)
		if check_pass: return True
	return False

def terminate():
	start.Terminate = True
	terminate.Terminate = True
	from os import system, listdir
	cmd = "rm {} {} {}".format(main.p_o, main.p_a, main.p_b)
	if system(cmd): raise Exception('fail to '+cmd)
	if len(listdir(init.fifodir))==0:
		cmd = "rm -r "+init.fifodir
		if system(cmd): raise Exception('fail to '+cmd)
def init():
	start.Terminate = True
#	start.exec_interactive = False
	terminate.Terminate = False

	start.ptr_int32_1 = (c_int32*1)()
	start.ptr_float_1 = (c_float*1)()

	from os import system, path
	init.fifodir = "/tmp/nle_win_fifo/"
	main.p_o = init.fifodir+'o' # pipe_observation
	main.p_a = init.fifodir+'a'
	main.p_b = init.fifodir+'b'
	if not path.isdir(init.fifodir):
		if system("mkdir "+init.fifodir):
			raise Exception('fail to mkdir '+init.fifodir)
	fifos = ''
	for fifo in (main.p_o, main.p_a, main.p_b):
		if not path.exists(fifo):
			fifos += ' '+fifo
	if len(fifos):
		if system("mkfifo"+fifos):
			raise Exception('fail to mkfifo in '+init.fifodir)

#def toggle_interactive():
#	start.exec_interactive = not start.exec_interactive

def start():
	from . import basic as p
	from .basic import Type
	frcv = frame()
	start.Terminate = False
	while not start.Terminate:
	#	frcv.f.head.head.type = Type['Ack']
		for _ in range(3): # 至多连续三次 cannot_* 错误后结束
			errNo = frcv.recv()
			check_pass = errno_check(frcv, errNo)
			if check_pass: break
			elif _ == 2:
				raise Exception("fail to recv")

		_ptr = (c_int32*1)(frcv.type())
		fsnd = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['Ack'])
		if not sendn(fsnd):
			raise Exception("fail to send Ack 1")
		if frcv.type() not in Type.values():
			raise Exception("Unknown type")
		if frcv.type() == Type['command']:
			cmd = bytes(frcv.f.ptr[0:len(frcv)])
			try:
				exec(cmd, globals())
				start.ptr_int32_1[0] = 0
			except:
				start.ptr_int32_1[0] = -1
				from traceback import print_exc
				print(print_exc())
			del cmd
			fsnd = frame(ptr=start.ptr_int32_1, ptr_len=sizeof(start.ptr_int32_1), len=sizeof(start.ptr_int32_1), type=Type['Ack'])
			if not sendn(fsnd):
				raise Exception("fail to send Ack 2")
		elif frcv.type() == Type['observation']: # 发送 observation，使用优化后的 c 函数
			try:
				p.send_obs(env.observation)
			except:
				raise Exception("fail to send obs")
		elif frcv.type() == Type['reward']:
			start.ptr_float_1[0] = env.reward
			fsnd = frame(ptr=start.ptr_float_1, ptr_len=sizeof(start.ptr_float_1), len=sizeof(start.ptr_float_1), type=frcv.type())
			if not sendn(fsnd):
				raise Exception("fail to send reward")
		elif frcv.type() == Type['done']:
			start.ptr_int32_1[0] = ~0 if env.done else 0
			fsnd = frame(ptr=start.ptr_int32_1, ptr_len=sizeof(start.ptr_int32_1), len=sizeof(start.ptr_int32_1), type=frcv.type())
			if not sendn(fsnd):
				raise Exception("fail to send done")
		elif frcv.type() == Type['info']:
			print(env.info)
			continue
		elif frcv.type() == Type['action']:
			action = cast(frcv.f.ptr, POINTER(c_int32))[0]
			env.step(action)
			p.send_obs(env.observation)
			start.ptr_float_1[0] = env.reward
			fsnd = frame(start.ptr_float_1, sizeof(start.ptr_float_1), sizeof(start.ptr_float_1), type=Type['reward'])
			if not errno_check(fsnd, fsnd.send()):
				raise Exception('fail to send reward')
			start.ptr_int32_1[0] = ~0 if env.done else 0
			fsnd = frame(start.ptr_int32_1, sizeof(start.ptr_int32_1), sizeof(start.ptr_int32_1), type=Type['done'])
			if not errno_check(fsnd, fsnd.send()):
				raise Exception('fail to send done')
		elif frcv.type() == Type['primitive_actions']:
			total_reward = 0
			try:
				for action in bytes(frcv.f.ptr[0:len(frcv)]):
					_, reward, _ = env.primitive_step(action)
					total_reward += reward
				start.ptr_int32_1[0] = 0
			except:
				start.ptr_int32_1[0] = -1
				from traceback import print_exc
				print(print_exc())
			fsnd = frame(ptr=start.ptr_int32_1, ptr_len=sizeof(start.ptr_int32_1), len=sizeof(start.ptr_int32_1), type=Type['Ack'])
			if not sendn(fsnd):
				raise Exception("fail to send Ack 2")
			p.send_obs(env.observation)
			start.ptr_float_1[0] = env.reward
			fsnd = frame(start.ptr_float_1, sizeof(start.ptr_float_1), sizeof(start.ptr_float_1), type=Type['reward'])
			if not errno_check(fsnd, fsnd.send()):
				raise Exception('fail to send reward')
			start.ptr_int32_1[0] = ~0 if env.done else 0
			fsnd = frame(start.ptr_int32_1, sizeof(start.ptr_int32_1), sizeof(start.ptr_int32_1), type=Type['done'])
			if not errno_check(fsnd, fsnd.send()):
				raise Exception('fail to send done')
		else:
			continue
	start.Terminate = True

def main():
	from . import basic as p
	init()
	cnt=0
	print("begin")
	while not terminate.Terminate:
		p.obs.open_pipe(main.p_o)
		p.bytes.openPipe(main.p_a, 0)
		p.bytes.openPipe(main.p_b, 1)
		if start.Terminate:
			start.Terminate = False
			print('start')
		else:
			print('restart', cnt)
		try:
			start()
		except:
			from traceback import print_exc
			print(print_exc())
			print('try reconnect')

		p.obs.close_pipe()
		p.bytes.closePipe(0)
		p.bytes.closePipe(1)
		start.Terminate = False
		print('end')
		cnt+=1
	start.Terminate = True
	print('Terminate')

if __name__ == '__main__':
	main()
