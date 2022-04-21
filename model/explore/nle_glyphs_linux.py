import gym

class GlyphCollection:
	def __init__(self, radius_y:int=0, radius_x:int=0, default_pixel:str=None):
		if radius_x<0: raise Exception('radius_x<0')
		if radius_y<0: raise Exception('radius_y<0')
		self.u, self.v = radius_y*2+1, radius_x*2+1
		self.Collection = {}
		self.default_char = ord(default_pixel[0]) if default_pixel else ord(' ')
	def Collect_glyph(self, color, char, glyph, Vstart:int, Vend:int, Hstart:int, Hend:int):
		shape = char.shape # (21, 79)
		Vstart, Vend = max(0, Vstart), min(Vend, shape[0])
		Hstart, Hend = max(0, Hstart), min(Hend, shape[1])
		keys = self.Collection.keys()
		for i in range(Vstart, Vend):
			for j in range(Hstart, Hend): # for every glyph in 21*79 glyphs
				if glyph[i][j] in keys:
					continue
				CHAR = [None]*(self.u*self.v)
				COLOR = [None]*(self.u*self.v)
				k = 0
				for y in range(i-self.u//2, i+self.u-self.u//2): # make stamp / icon
					for x in range(j-self.v//2, j+self.v-self.v//2):
						CHAR[k], COLOR[k] = (char[y][x], color[y][x]) if not any((x<0, y<0, shape[0]<=y, shape[1]<=x)) else (self.default_char, 0)
						k += 1
				self.Collection[glyph[i][j]] = (CHAR, COLOR, i, j)
				keys = self.Collection.keys()
	def print_Collection(self):
		print('glyph_hex @(coord) [char#color]')
		for key, value in self.Collection.items():
			pos = len(value[0])//2 # center
			print('%4X @(%02d, %02d) [%s]'%(
					key,
					value[2], value[3],
					'\033[38;5;%dm%c#%02X\033[0m'%(value[1][pos], value[0][pos], value[1][pos])
				)
			)
			print('\t'+'┏' + '━'*self.v + '┓')
			s = '' # ┏━┓┛┗┃
			for pos, (char, color) in enumerate(zip(value[0], value[1])):
				s += '\033[38;5;%dm%c'%(color, char)
				if not (pos+1)%self.v:
					print('\t'+'┃'+s+'\033[0m┃')
					s = ''
			print('\t'+'┗' + '━'*self.v + '┛')

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
def play(env:gym.Env, Collection:GlyphCollection=None):
	keymap = {
		'k': 0, 'l': 1, 'j': 2, 'h': 3, 'u': 4, 'n': 5, 'b': 6, 'y': 7, # CompassDirection, numboard
		'K': 8, 'L': 9, 'J': 10, 'H': 11, 'U': 12, 'N': 13, 'B': 14, 'Y': 15, # CompassDirectionLonger
		'<': 16, '>': 17, '.': 18, # UP, DOWN, WAIT
		'\r':19, # MORE
		'\033': 38, # <ESC>
		' ': 107,# SPACE
	}
	obs, rwd, dn, info = env.reset(), 0., False, {}
	env.render()
	while not dn:
		if Collection is not None:
			y, x = obs['tty_cursor']; y -= 1 # character coord
			Collection.Collect_glyph(obs['colors'], obs['chars'], obs['glyphs'], y-5, y+6, x-5, x+6)
		s = Getch()
		if s=='S':
			break
		try:
			action = keymap[s[0]]
		except:
			print('Invalid key "{}"'.format(s))
			continue
		obs, rwd, dn, info = env.step(action)
		env.render()
	env.close()
	return obs, rwd, dn, info

def Cut(c): # char
	Vstart, Vend, Hstart, Hend = (0,)*4
	space = ord(' ')
	for i in range(0, len(c), 1):
		if sum(c[i])-space*len(c[i]):
			Vstart = i
			break
	for i in range(len(c), 0, -1):
		if (sum(c[i-1]))-space*len(c[i-1]):
			Vend = i
			break
	c = c.transpose()
	for i in range(0, len(c), 1):
		if sum(c[i])-space*len(c[i]):
			Hstart = i
			break
	for i in range(len(c), 0, -1):
		if (sum(c[i-1]))-space*len(c[i-1]):
			Hend = i
			break
	c = c.transpose()
	for i in range(Vstart, Vend):
		s = ''
		for j in range(Hstart, Hend):
			s += chr(c[i][j])
		print(s)
	print('[{}:{}][{}:{}]'.format(Vstart, Vend, Hstart, Hend))
	return Vstart, Vend, Hstart, Hend

def main():
	import nle
	env = gym.make('NetHackChallenge-v0', savedir=None)

	Instances = GlyphCollection(1, 2, ' ')
	u, v = 3, 5 # icon width, height
	obs, rwd, dn, info = play(env, Instances)

	Vstart, Vend, Hstart, Hend = Cut(obs['chars'])

	color = obs['colors']
	char = obs['chars']
	glyph = obs['glyphs']
	Instances.Collect_glyph(color, char, glyph, Vstart, Vend, Hstart, Hend)

	Instances.print_Collection()

if __name__ == '__main__':
	main()
