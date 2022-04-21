from ..LibFrom import From, Import
from ctypes import Structure, c_int32, c_int64, c_short, c_uint8, byref, create_string_buffer
from os import path
dirname = path.dirname(path.abspath(__file__))
lib:str = '.\\obs\\lib.dll'
lib = dirname + '\\..\\lib\\' + lib
del path

class observation(Structure):
	_fields_ = [
		('glyphs'             , c_short * 79 * 21),
		('chars'              , c_uint8 * 79 * 21),
		('colors'             , c_uint8 * 79 * 21),
		('specials'           , c_uint8 * 79 * 21),
		('blstats'            , c_int64 * 26),
		('message'            , c_uint8 * 256),
		('inv_glyphs'         , c_short * 55),
		('inv_strs'           , c_uint8 * 80 * 55),
		('inv_letters'        , c_uint8 * 55),
		('inv_oclasses'       , c_uint8 * 55),
	#	('screen_descriptions', c_uint8 * 80 * 79 * 21),
		('tty_chars'          , c_uint8 * 80 * 24),
		('tty_colors'         , c_uint8 * 80 * 24),
		('tty_cursor'         , c_uint8 * 2),
		('misc'               , c_int32 * 3)
	]
	def __init__(self, *args, **kw) -> None:
		self.glyphs       = (c_short * 79 * 21)()
		self.chars        = (c_uint8 * 79 * 21)()
		self.colors       = (c_uint8 * 79 * 21)()
		self.specials     = (c_uint8 * 79 * 21)()
		self.blstats      = (c_int64 * 26)()
		self.message      = (c_uint8 * 256)()
		self.inv_glyphs   = (c_short * 55)()
		self.inv_strs     = (c_uint8 * 80 * 55)()
		self.inv_letters  = (c_uint8 * 55)()
		self.inv_oclasses = (c_uint8 * 55)()
		self.tty_chars    = (c_uint8 * 80 * 24)()
		self.tty_colors   = (c_uint8 * 80 * 24)()
		self.tty_cursor   = (c_uint8 * 2)()
		self.misc         = (c_int32 * 3)()
		super().__init__(*args, **kw)

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
