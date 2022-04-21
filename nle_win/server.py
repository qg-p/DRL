# run this file with python with nethack learning environment in linux (WSL)
from ctypes import POINTER, c_float, c_int32, sizeof, cast
from .basic import frame

class NLEnv():
	def __init__(self, env_id='NetHackChallenge-v0', *args, **kwargs) -> None:
		import nle
		from .basic.common import observation, reward, done, info
		self.env = nle.env.gym.make(env_id, *args, **kwargs)
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
		_, self.done = self.env.env.step(action) # (strange obs, reward)
		return _, self.done
	def update_observation(self):
		self.observation=self.env._get_observation(self.env.env._obs)
		return self.observation

env = NLEnv(character='@', savedir=None)

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
	from os import system
	if system("rm -r "+init.fifodir): raise Exception('fail to rm -r '+init.fifodir)
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
		if system("mkfifo %s %s %s" % (main.p_o, main.p_a, main.p_b)):
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
			try:
				for action in bytes(frcv.f.ptr[0:len(frcv)]):
					env.primitive_step(action)
				start.ptr_int32_1[0] = 0
			except:
				start.ptr_int32_1[0] = -1
				from traceback import print_exc
				print(print_exc())
			fsnd = frame(ptr=start.ptr_int32_1, ptr_len=sizeof(start.ptr_int32_1), len=sizeof(start.ptr_int32_1), type=Type['Ack'])
			if not sendn(fsnd):
				raise Exception("fail to send Ack 2")
			env.update_observation()
			p.send_obs(env.observation)
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
