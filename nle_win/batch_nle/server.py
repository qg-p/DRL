class env_batch:
	def __init__(self, batch_size:int, env_id='NetHackChallenge-v0', *args, **kwargs):
		from .batch import batch_frame
		from ..server import NLEnv
		assert batch_size > 0
		self.env = tuple(NLEnv(env_id, *args, **kwargs) for _ in range(batch_size))
		self.fsnd = batch_frame(batch_size)

	def step(self, action_batch:list):
		assert len(action_batch)==self.fsnd.batch_size
		from ctypes import sizeof, memmove
		for i, (env, action) in enumerate(zip(self.env, action_batch)):
			unit = self.fsnd.batch[i]
			if unit.done or action == 255: # -1 in byte
				observation, reward, done = env.reset(), 0., False
			else:
				observation, reward, done = env.primitive_step(action)
			for (k, v) in observation.items():
				t = getattr(unit.obs, k)
				memmove(t, v.ctypes._as_parameter_, sizeof(t))
			unit.done = 1 if done else 0
			unit.reward = reward
		return self.fsnd.batch

	def close(self):
		for env in self.env:
			env.close()
	def render(self, i:int, mode='human'):
		return self.env[i].render(mode)

env:env_batch = None#(1, character='@', savedir=None, penalty_step=-0.01)

''' TODO
utilize:
	command
	receive batch_action, execute (step & reset)
	genereate batch, send_batch
	ack
'''

def terminate():
	start.Terminate = True
	terminate.Terminate = True
	from os import system, listdir
	cmd = "rm {} {}".format(main.p_a, main.p_b)
	if system(cmd): raise Exception('fail to '+cmd)
	if listdir(init.fifodir)==0:
		cmd = "rm -r "+init.fifodir
		if system(cmd): raise Exception('fail to '+cmd)

def init():
	start.Terminate = True
	terminate.Terminate = False

	from os import system, path
	init.fifodir = "/tmp/nle_win_fifo/"
	main.p_a = init.fifodir+'_batch_a_'
	main.p_b = init.fifodir+'_batch_b_'
	if not path.isdir(init.fifodir):
		if system("mkdir "+init.fifodir):
			raise Exception('fail to mkdir '+init.fifodir)
	fifos = ''
	for fifo in (main.p_a, main.p_b):
		if not path.exists(fifo):
			fifos += ' '+fifo
	if len(fifos):
		if system("mkfifo"+fifo):
			raise Exception('fail to mkfifo in '+init.fifodir)

def start():
	from ..server import errno_check, sendn
	from ..basic import frame, Type
	frcv = frame()
	start.Terminate = False
	global env
	while not start.Terminate:
		for _ in range(3):
			errNo = frcv.recv()
			check_pass = errno_check(frcv, errNo)
			if check_pass: break
			elif _ == 2:
				raise Exception("fail to recv")

		from ctypes import sizeof, c_int32
		_ptr = (c_int32*1)(frcv.type())
		fack = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['Ack'])
		if not sendn(fack):
			raise Exception("fail to send Ack 1")
		if frcv.type() not in Type.values():
			raise Exception("Unknown type")
		if frcv.type() == Type['command']:
			# 1. exec | extract cmd, execute cmd, process exception
			# 2. send | gen fack representing exception, send ack 2
			cmd = bytes(frcv.f.ptr[0:len(frcv)])
			try:
				exec(cmd, globals())
				_ptr[0] = 0
			except:
				_ptr[0] = -1
				from traceback import print_exc
				print(print_exc())
			del cmd
			fack = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['Ack'])
			if not sendn(fack):
				raise Exception("fail to send Ack 2")
		elif frcv.type() == Type['action_batch']: # int32*batch_size
			# 1. exec | env.step(action batch), write frame_batch, process exception
			# 2. send | send ack(-1) if any exception is raised, else send ack(0)
			# 3. send | send frame_batch if ack(0) is sent
			# increase frcv.ptr.capacity if it is too small to receive 4*batch_size bytes of actions
			action_batch = [*bytes(frcv.f.ptr[0:len(frcv)])]
			try:
				env.step(action_batch)
				_ptr[0] = 0
			except:
				_ptr[0] = -1
				from traceback import print_exc
				print(print_exc())
			del action_batch
			fack = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['Ack'])
			if not sendn(fack):
				raise Exception("fail to send Ack 2")
			if not errno_check(env.fsnd, env.fsnd.send()):
				raise Exception("fail to send batch result")
def main():
	from .. import basic as p
	init()
	cnt=0
	print("begin (batch)")
	while not terminate.Terminate:
		p.bytes.openPipe(main.p_a, 0)
		p.bytes.openPipe(main.p_b, 1)
		if start.Terminate:
			start.Terminate = False
			print('start')
		else:
			print('restart', cnt)
		try:
			start()
			p.bytes.frame(ptr_len=0).recv() # block
		except:
			from traceback import print_exc
			print(print_exc())

		p.bytes.closePipe(0)
		p.bytes.closePipe(1)
		start.Terminate = False
		print('end')
		cnt+=1
	start.Terminate = True
	print('Terminate (batch)')