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

class Frame_header(Structure):
	_fields_ = [
		('l', c_uint32),
		('head', Pkg_header)
	]

class Frame(Structure):
	_fields_ = [
		('ptr', c_ubyte_p),
		('ptr_len', c_uint32),
		('head', Frame_header)
	]

c_varframe = POINTER(Frame)

def BUFFERlen()->c_uint32:
	return 4096

from ctypes import cast
def getBUFFER()->c_ubyte_p:
	return cast(getBUFFER.BUFFER, c_ubyte_p)
if not hasattr(getBUFFER, 'BUFFER'):
	getBUFFER.BUFFER = (c_ubyte*BUFFERlen())()
