from ..client import ack_check, sendn, recv_ack

def EXEC(cmd:str):
	from ctypes import c_ubyte, sizeof
	from ..basic import frame, Type
	_ptr=(c_ubyte*len(cmd))(*(cmd.encode())) # ?10 -> \x0d0a
	fsnd = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['command'])
	a = sendn(fsnd) # whether it is sent
	if not a: raise Exception('fail to send')
	a, b, c, content, fack = recv_ack(fsnd.type())
	if fack.type() == Type['errNo']:
		print('Error: command string is too long: %d > %d' % (len(cmd), content))
		raise Exception('too long for recv')
	a, b, c, _, fack = recv_ack(0)
	if not (a and b and c):
		raise Exception('Server exec({}) failed'.format(cmd))

class batch:
	def __init__(self, batch_size:int, batch_env_param:str='character="@", savedir=None, penalty_step=-0.01'):
		from .batch import batch_frame
		EXEC('env=env_batch({},{})'.format(batch_size, batch_env_param))

		self.frcv = batch_frame(batch_size)
		from ..basic.common import frame, Type
		from ..basic.core.bytes.classes import c_ubyte_p
		from ctypes import c_byte, cast, sizeof
		ptr = (c_byte*batch_size)()
		self.fsnd = frame(cast(ptr, c_ubyte_p), sizeof(ptr), sizeof(ptr), Type['action_batch'])
		self.fsnd_ptr = ptr
	def reset(self):
		return self.step([-1]*self.frcv.batch_size)
	from typing import List
	def step(self, action_batch:List[int]):
		'''
		action_batch[i] == -1 for reset
		uses primitive action
		send in bytes form
		'''
		for i, action in enumerate(action_batch):
			self.fsnd_ptr[i] = action
		if not sendn(self.fsnd): raise Exception('fail to send')
		ack_check(self.fsnd.type())
		ack_check(0) # whether step(action) raise no exception
		self.frcv.recv()
		return self.frcv.batch[0:self.frcv.batch_size]

from .. import basic as p

def connect(
	p_a:str='//wsl$/Ubuntu/tmp/nle_win_fifo/_batch_a_',
	p_b:str='//wsl$/Ubuntu/tmp/nle_win_fifo/_batch_b_'
):
	p.bytes.openPipe(p_a, 1)
	p.bytes.openPipe(p_b, 0)

def disconnect():
	EXEC('start.Terminate = True')
	p.bytes.closePipe(1)
	p.bytes.closePipe(0)

def terminate():
	EXEC('terminate()')

if __name__ == '__main__':
	connect()
	disconnect()