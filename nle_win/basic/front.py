from .core import obs as OBS, bytes as B
from .common import *

if __name__ == '__main__':
	from ctypes import cast, c_char_p
	obs=OBS.observation()
	OBS.recv(obs)
	print(cast(obs.message, c_char_p).value)