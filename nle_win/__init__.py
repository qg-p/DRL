from sys import platform
if platform == 'linux':
	from .server import frame, main, env
elif platform == 'win32':
	from .client import frame, getobs, getreward, getdone, Exec, connect, step, InteractiveEXEC, step_primitive, disconnect
else:
	print(platform, 'is not supported.')
from . import basic
from . import batch_nle
del platform
