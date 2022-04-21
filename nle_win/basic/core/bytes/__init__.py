from sys import platform
if platform == 'linux':
	from .wsl import Frame, new_ID, BUFFERlen, getBUFFER, send, recv, openPipe, closePipe, fread, fwrite
elif platform == 'win32':
	from .win import Frame, new_ID, BUFFERlen, getBUFFER, send, recv, openPipe, closePipe, fread, fwrite
else:
	print(platform, 'is not supported.')
del platform

class frame():
	from ctypes import c_uint32
	def __init__(self,
		ptr=getBUFFER(), ptr_len:c_uint32=BUFFERlen(), len:c_uint32=0,
		type:c_uint32=0, # bytes
		package_offset:c_uint32=0, package_length:c_uint32=0, package_ID:c_uint32=0 # pkg_ID=0 means use no package
	):
		from ctypes import cast
		from .classes import c_ubyte_p
		f = Frame()
		f.ptr = cast(ptr, c_ubyte_p)
		f.ptr_len = ptr_len
		f.head.l = len
		f.head.head.type = type
		f.head.head.pkg_offset = package_offset
		f.head.head.pkg_len = package_length
		f.head.head.pkg_ID = package_ID
		self.f = f

	def send(self)->int:
		from ctypes import pointer
		return send(pointer(self.f))

	def recv(self)->int:
		from ctypes import pointer
		return recv(pointer(self.f))
	
	def type(self)->int:
		return self.f.head.head.type

	def __len__(self):
		return int(self.f.head.l)

#	def __sizeof__(self):
#		return self.f.__sizeof__() + self.f.ptr_len

	def __iter__(self):
		class frame_iterator:
			from ctypes import c_ubyte
			def __init__(self, F:frame):
				self.f = F.f
				self.pos = 0
			def __next__(self)->c_ubyte:
				if self.pos >= self.f.head.l:
					raise StopIteration()
				return self.f.ptr[self.pos]
		return frame_iterator(self)
