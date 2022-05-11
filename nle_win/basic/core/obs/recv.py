from ..LibFrom import From, Import
from ctypes import byref, create_string_buffer
from os import path
dirname = path.dirname(path.abspath(__file__))
lib:str = '.\\obs\\lib.dll'
lib = dirname + '\\..\\lib\\' + lib
del path

from .observation import observation

@From(lib)
@Import('recv_obs')
def recv(self, obs:observation):#->c_int:
	return self.recv_obs(byref(obs))

@From(lib)
@Import('closePipe')
def close_pipe(self):
	return self.closePipe()

@From(lib)
@Import('openPipe')
def open_pipe(self, name:str):
	string=create_string_buffer(name.encode(), len(name)+1)
	return self.openPipe(string)
