from sys import platform
if platform == 'linux':
	from .server import main, env
elif platform == 'win32':
	from .client import connect, disconnect, EXEC, batch
else:
	print(platform, 'is not supported.')
del platform
