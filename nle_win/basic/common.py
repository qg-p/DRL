from ctypes import c_ubyte
from .core.bytes import frame

Type={
	'bytes':0,
	'command':1,
	'observation':2, 'reward':3, 'done':4, 'info':5,
	'action':6,
	'errNo':7, 'Ack':8,
	'primitive_actions':9,
	'observation_batch':10,
	'action_batch':11
}
ErrNo={
	'other':-1, 'success':0, 'warn':1,
	'cannot_open':2, 'cannot_read':3, 'cannot_write':4,
	'too_long':5
}

# def get():
# 	f = frame()
# 	errNo = f.recv()
# 	return f, errNo

# def post(f:frame):
# 	return f.send()

# def post_cmd(cmd:str):
# 	f = frame(ptr=c_ubyte*len(cmd), len=len(cmd), type=Type['command'])
# 	return post(f)

observation:dict={}
reward:float=0.0
done:bool=False
info:dict={} #?

# def automata():
# 	frcv = frame()
# 	while(1):
# 		errNo = frcv.recv()

# 		if errNo in (ErrNo['success'], ErrNo['warn']):
# 			pass
# 		elif errNo in (ErrNo['cannot_open'], ErrNo['cannot_read'], ErrNo['cannot_write'], ErrNo['too_long']):
# 			break
# 		else:
# 			from sys import _getframe
# 			print('%s(%d), %s: Error: unknown errNo' % (__file__, _getframe().f_lineno, automata.__name__))
# 			break

# 		if frcv.type() == Type['command']:
# 			cmd = bytes(frcv.f.ptr[0:len(frcv)])
# 			exec(cmd)
# 			del cmd
# 		elif frcv.type() in {Type['bytes'], Type['observation'], Type['reward'], Type['done'], Type['info']}:
# 			break
# 		else:
# 			from sys import _getframe
# 			print('%s(%d), %s: Error: unknown type' % (__file__, _getframe().f_lineno, automata.__name__))
# 			break

# 	return frcv, errNo
