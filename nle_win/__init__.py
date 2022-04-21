from sys import platform
if platform == 'linux':
	from .server import frame, main, env
	from . import basic
elif platform == 'win32':
	from .client import frame, getobs, getreward, getdone, Exec, connect, step, InteractiveEXEC, step_primitive, disconnect
	from . import basic
else:
	print(platform, 'is not supported.')
del platform
