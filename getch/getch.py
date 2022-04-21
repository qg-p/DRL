from sys import platform
if platform == 'linux':
	def Getch(): # Unix
		import sys, tty, termios
		fd = sys.stdin.fileno()
		init_settings = termios.tcgetattr(fd)
		try:
			tty.setraw(fd)
			ch = sys.stdin.read(1)
		finally:
			termios.tcsetattr(fd, termios.TCSADRAIN, init_settings)
		return ch
elif platform == 'win32':
	def Getch():
		from msvcrt import getch
		return getch()
else:
	print(platform, 'is not supported')
del platform
