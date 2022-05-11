from ctypes import c_int32, c_ubyte, c_uint32, POINTER, Structure
c_ubyte_p = POINTER(c_ubyte)

# type:
# enum{
# 	bytes=0, // send a vector
# 	observation, reward, done, info,
# 	(...,)
# 	command
# }
class Pkg_header(Structure):
	_fields_ = [
		('type', c_int32),
		('pkg_offset', c_uint32),
		('pkg_len', c_uint32),
		('pkg_ID', c_uint32)
	]
	class template:
		type = c_int32()
		pkg_offset = c_uint32()
		pkg_len = c_uint32()
		pkg_ID = c_uint32()
	def __init__(self, *args, **kwargs):
		self.type = Pkg_header.template.type
		self.pkg_offset = Pkg_header.template.pkg_offset
		self.pkg_len = Pkg_header.template.pkg_len
		self.pkg_ID = Pkg_header.template.pkg_ID
		super().__init__(*args, **kwargs)

class Frame_header(Structure):
	_fields_ = [
		('l', c_uint32),
		('head', Pkg_header)
	]
	class template:
		l = c_uint32()
		head = Pkg_header()
	def __init__(self, *args, **kwargs):
		self.l = Frame_header.template.l
		self.head = Frame_header.template.head
		super().__init__(*args, **kwargs)

class Frame(Structure):
	_fields_ = [
		('ptr', c_ubyte_p),
		('ptr_len', c_uint32),
		('head', Frame_header)
	]
	class template:
		ptr = c_ubyte_p()
		ptr_len = c_uint32()
		head = Frame_header()
	def __init__(self, *args, **kwargs):
		self.ptr = Frame.template.ptr
		self.ptr_len = Frame.template.ptr_len
		self.head = Frame.template.head
		super().__init__(*args, **kwargs)

c_varframe = POINTER(Frame)

def BUFFERlen()->c_uint32:
	return 4096

from ctypes import cast
def getBUFFER()->c_ubyte_p:
	return cast(getBUFFER.BUFFER, c_ubyte_p)
if not hasattr(getBUFFER, 'BUFFER'):
	getBUFFER.BUFFER = (c_ubyte*BUFFERlen())()
