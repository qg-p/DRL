from ..LibFrom import From, Import
from ctypes import c_int32, cast, POINTER, c_uint8, c_short, c_int64, create_string_buffer
from os import path
dirname = path.dirname(path.abspath(__file__))
lib:str = './obs/lib.so'
lib = dirname + '/../lib/' + lib
del path

@From(lib)
@Import('send_obs')
def send(self, glyphs, chars, colors, specials, blstats, message, inv_glyphs, inv_strs, inv_letters, inv_oclasses, tty_chars, tty_colors, tty_cursor, misc):
	self.send_obs(
		cast(glyphs              .ctypes._as_parameter_, POINTER(c_short)),
		cast(chars               .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(colors              .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(specials            .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(blstats             .ctypes._as_parameter_, POINTER(c_int64)),
		cast(message             .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(inv_glyphs          .ctypes._as_parameter_, POINTER(c_short)),
		cast(inv_strs            .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(inv_letters         .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(inv_oclasses        .ctypes._as_parameter_, POINTER(c_uint8)),
	#	cast(screen_descriptions .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(tty_chars           .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(tty_colors          .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(tty_cursor          .ctypes._as_parameter_, POINTER(c_uint8)),
		cast(misc                .ctypes._as_parameter_, POINTER(c_int32))
	)

@From(lib)
@Import('closePipe')
def close_pipe(self):
	return self.closePipe()

@From(lib)
@Import('openPipe')
def open_pipe(self, name:str):
	string=create_string_buffer(name.encode(), len(name)+1)
	return self.openPipe(string)
