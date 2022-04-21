from ctypes import POINTER, c_float, c_ubyte, cast, c_int32, sizeof
from . import basic as p
from .basic import frame, Type, ErrNo

# return: whether sent/received successfully
def errno_check(f:frame, errNo:int)->bool:
	if errNo in (ErrNo['success'], ErrNo['warn']):
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

# return: [bool(recv), bool(is_valid_ack), bool(exec_no_exception), frcv]
def Exec(cmd:str):
	from ctypes import c_ubyte
	_ptr=(c_ubyte*len(cmd))(*(cmd.encode())) # ?10 -> \x0d0a
	fsnd = frame(ptr=_ptr, ptr_len=sizeof(_ptr), len=sizeof(_ptr), type=Type['command'])
	a = sendn(fsnd) # whether it is sent
	if not a: raise Exception('fail to send')

	a, b, c, content, fack = recv_ack(fsnd.type())
	if fack.type() == Type['errNo']:
		print('Error: command string is too long: %d > %d' % (len(cmd), content))
		raise Exception('too long for recv')
	a, b, c, _, fack = recv_ack(0) # bool(_) == c
	return a, b, c, fack

def InteractiveEXEC():
	InteractiveEXEC.n = False
	InteractiveEXEC.CMD = ''
	print('>>>')
	while True:
		InteractiveEXEC.line = input()
		if not InteractiveEXEC.n:
			InteractiveEXEC.CMD = InteractiveEXEC.line
		else:
			InteractiveEXEC.CMD += '\n' + InteractiveEXEC.line
		if InteractiveEXEC.CMD == "exit": break
		if len(InteractiveEXEC.line) == 0: InteractiveEXEC.n = False # empty line
		elif InteractiveEXEC.line[len(InteractiveEXEC.line)-1] == ':': InteractiveEXEC.n = True
		if not InteractiveEXEC.n:
			a, b, c, _ = Exec(InteractiveEXEC.CMD) # notice that empty line also trigger a send (20+24+24)
			if not a: print('ack2 recv fail')
			if not b: print('frcv is not ack')
			if not c: print('Exception')
			print('>>>')

def recv_ack(_type:int):
	_ptr = (c_int32*1)()
	fack = frame(_ptr, sizeof(_ptr))
	errno = fack.recv()
	a:bool = errno_check(fack, errno) # whether the respose is received

	b:bool = fack.type() == Type['Ack'] # whether command is received
	content = int(cast(fack.f.ptr, POINTER(c_int32))[0])
	c:bool = content == _type if _type is not None else True # whether command is received as command

	return a, b, c, content, fack

def ack_check(_TYPE:int):
	a, b, _, content, fack = recv_ack(_TYPE)
	if not a:
		raise Exception('errno check fail')
	if not b:
		raise Exception('not ack. content: %s' % (content))
	if fack.type() == Type['errNo']:
		raise Exception('fsnd too long') # never reach
	return fack

def getobs()->p.obs.observation:
	if not sendn(getobs.fsnd): raise Exception('fail to send')
	ack_check(getobs.fsnd.type())
	p.obs.recv(getobs.obs)
	return getobs.obs
getobs.fsnd = frame(0, 0, type=Type['observation'])
getobs.obs = p.obs.observation()

def getreward():
	if not sendn(getreward.fsnd): raise Exception('fail to send')
	ack_check(getreward.fsnd.type())
	getreward.frcv.recv()
	getreward.reward = float(cast(getreward.frcv.f.ptr, POINTER(c_float))[0])
	return getreward.reward
getreward.frcv = frame((c_float*1)(), sizeof((c_float*1)()))
getreward.fsnd = frame(0, 0, type=Type['reward'])
getreward.reward = 0.0

def getdone():
	if not sendn(getdone.fsnd): raise Exception('fail to send')
	ack_check(getreward.fsnd.type())
	getdone.frcv.recv()
	getdone.done = cast(getdone.frcv.f.ptr, POINTER(c_int32))[0]!=0
	return getdone.done
getdone.frcv = frame((c_int32*1)(), sizeof((c_int32*1)()))
getdone.fsnd = frame(0, 0, type=Type['done'])
getdone.done = False

# execute multiple actions efficiently, returns obs, reward, done
def step_primitive(action_seq:bytes):
	step_primitive.fsnd.f.ptr = cast(action_seq, POINTER(c_ubyte))
	step_primitive.fsnd.f.head.l = len(action_seq)
	if not sendn(step_primitive.fsnd): raise Exception('fail to send')
	ack_check(step_primitive.fsnd.type())
	a, b, c, _, _ = recv_ack(0) # bool(_) == c
	if not (a and b and c):
		raise Exception('fail to step along'+action_seq.decode())
	p.obs.recv(getobs.obs)
	getdone.frcv.recv()
	getdone.done = cast(getdone.frcv.f.ptr, POINTER(c_int32))[0]!=0
	return getobs.obs, getdone.done
step_primitive.frcv = frame((c_int32*1)(), sizeof((c_int32*1)()))
step_primitive.fsnd = frame(0, 0, type=Type['primitive_actions'])
step_primitive.done = False

# obs, reward, done
def step(action:int):
	step.action[0] = action
	if not sendn(step.fsnd): raise Exception('fail to send')
	ack_check(step.fsnd.type())
	p.obs.recv(getobs.obs)
	getreward.frcv.recv()
	getreward.reward = float(cast(getreward.frcv.f.ptr, POINTER(c_float))[0])
	getdone.frcv.recv()
	getdone.done = cast(getdone.frcv.f.ptr, POINTER(c_int32))[0]!=0
	return getobs.obs, getreward.reward, getdone.done
ptr = (c_int32*1)()
step.frcv = frame(ptr, sizeof(ptr))
step.action = (c_int32*1)()
step.fsnd = frame(step.action, len=sizeof(step.action), type=Type['action'])

def connect(
	p_o:str='//wsl$/Ubuntu/tmp/nle_win_fifo/o',
	p_a:str='//wsl$/Ubuntu/tmp/nle_win_fifo/a',
	p_b:str='//wsl$/Ubuntu/tmp/nle_win_fifo/b'
):
	p.obs.open_pipe(p_o)
	p.bytes.openPipe(p_a, 1)
	p.bytes.openPipe(p_b, 0)

def disconnect():
	p.obs.close_pipe()
	Exec('start.Terminate = True')
	p.bytes.closePipe(1)
	p.bytes.closePipe(0)

if __name__ == '__main__':
	a, b, c, _ = Exec('import nle; print(dir(nle)); del nle;')
	if not (a and b and c):
		print('Exec 1 fail:', a, b, c)
	a, b, c, _ = Exec('raise Exception("test")')
	if not (a and b and not c):
		print('Exec 2 fail:', a, b, c)
	a, b, c, _ = Exec('start.Terminate = True')
	if not (a and b and c):
		print('Exec 3 fail:', a, b, c)
	a, b, c, _ = Exec('terminate()')
	if not (a and b and c):
		print('Exec 4 fail:', a, b, c)
