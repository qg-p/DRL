from sys import platform
if platform == 'linux':
	from .back import  frame, get, post, post_cmd, automata, Type, ErrNo, B as bytes, OBS as obs, send_obs
elif platform == 'win32':
	from .front import frame, get, post, post_cmd, automata, Type, ErrNo, B as bytes, OBS as obs
else:
	print(platform, 'is not supported.')
del platform