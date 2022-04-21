from sys import platform
if platform == 'linux':
	from .send import send, open_pipe, close_pipe
elif platform == 'win32':
	from .recv import recv, open_pipe, close_pipe, observation
else:
	print(platform, 'is not supported.')
del platform

if __name__ == '__main__':
	print('python3 -m *.*.*(.py)')