from ..LibFrom import *
from .classes import Frame, c_varframe, c_ubyte_p, BUFFERlen, getBUFFER
from ctypes import c_int32, c_uint32, c_void_p, cast, create_string_buffer
from os import path
dirname = path.dirname(path.abspath(__file__))
lib:str = './bytes/bin/win.dll'
lib = dirname + '/../lib/' + lib
del path

@From(lib)
@Import('new_ID')
def new_ID(self)->c_uint32:
	return self.new_ID()

# return:
# enum{	_____=-1, success=0, ____=1,
# 		cannot_open, ___________, cannot_write,
# 		________,}
@From(lib)
@Import('send')
def send(self, F:c_varframe)->c_uint32:
	return self.send(F)

# return:
# enum{	_____=-1, success=0, ____=1,
# 		cannot_open, cannot_read, ____________,
# 		too_long,}
@From(lib)
@Import('recv')
def recv(self, F:c_varframe)->c_uint32:
	return self.recv(F)

# file_index:
# 	0: write
# 	1: read
# filename:
# 	"": default
# 	else: use this file
@From(lib)
@Import('openPipe')
def openPipe(self, filename:str, file_index:c_int32)->None:
	filename = create_string_buffer(filename.encode(), len(filename)+1)
	self.openPipe(filename, file_index)

# file_index:
# 	0: write
# 	1: read
@From(lib)
@Import('closePipe')
def closePipe(self, file_index:c_int32)->None:
	self.closePipe(file_index)

# read n*size bytes from r pipe to ptr
# mention:
# 	it is not safe using fread directly without getw(fp[1])
# 	because 'fread(void*, int, int, FILE*)->int' in mingw-w64 doesn't block when it is empty
@From(lib)
@Import('freadSafe')
def fread(self, ptr:c_ubyte_p, size:c_int32, n:c_int32)->c_uint32:
	return self.fread(cast(ptr, c_void_p), size, n)

# write n*size bytes to w pipe from ptr
@From(lib)
@Import('fwriteSafe')
def fwrite(self, ptr:c_ubyte_p, size:c_int32, n:c_int32)->c_uint32:
	return self.fwrite(cast(ptr, c_void_p), size, n)
