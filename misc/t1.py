if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

from nle_win.client import step_primitive


KEYMAP = {
	'k': 0, 'l': 1, 'j': 2, 'h': 3, 'u': 4, 'n': 5, 'b': 6, 'y': 7, # CompassDirection, numboard
	'K': 8, 'L': 9, 'J': 10, 'H': 11, 'U': 12, 'N': 13, 'B': 14, 'Y': 15, # CompassDirectionLonger
	'<': 16, '>': 17, '.': 18, # UP, DOWN, WAIT
	'\r':19, # MORE
# Command.
	'a': 24, # APPLY
	'@': 26, # AUTOPICKUP
	'C': 27, # CALL
	'Z': 28, # CAST a spell
	'c': 30, # CLOSE
	'd': 33, # DROP
	'D': 34, # DROPTYPE
	'e': 35, # EAT
	'E': 36, # ENGRAVE
	'\x1b': 38, # <ESC>
	'F': 39, # FIGHT
	'f': 40, # FIRE
	';': 42, # GLANCE
	'V': 43, # HISTORY
	'i': 44, # INVENTORY
	'I': 45, # INVERTORYTYPE
	'\\':49, # KNOWN
	'`': 50, # KNOWNCLASS
	':': 51, # LOOK what is under @
	'm': 54, 'g': 54, # MOVE
	'M': 55, 'G': 55, # MOVEFAR
	'o': 57, # OPEN a container on the floor
	'O': 58, # OPTIONS
	'p': 60, # PAY
	',': 61, # PICKUP
	'P': 63, # PUTON
	'q': 64, # QUAFF
	'Q': 66, # QUIVER
	'r': 67, # READ
	'R': 69, # REMOVE amulet, ring, eyewear
	'S': 74, # SAVE
	's': 75, # SEARCH
	'*': 76, # SEE ALL
	'"': 77, # SEE AMULET
	'[': 78, # SEE ARMOR
	'$': 79, # SEE GOLD
	'=': 80, # SEE RINGS
	'+': 81, # SEE SPELLS
	'(': 82, # SEE TOOLS
	'^': 83, # SEE TRAP
	')': 84, # SEE WEAPON
	'!': 85, # SHELL
	'x': 87, # SWAP (two weapons)
	'T': 88, # TAKEOFF
	'A': 89, # TAKE OFF ALL
	't': 91, # THROW
	'_': 93, # TRAVEL
	'X': 95, # TWOWEAPON
	'v': 98, # VERSIONSHORT
	'W': 99, # WEAR armor, shield, rings, etc.
	'&': 100,# WHAT (a key) DOES
	'/': 101,# WHAT IS (...)
	'w': 102,# WIELD weapon
	'z': 104,# ZAP a wand
# TextCharacters.
	'+': 105,# PLUS
	'-': 106,# MINUS
	' ': 107,# SPACE
	"'": 108,# APOS
	'"': 109,# QUOTE
	'0': 110,# NUM_0
	'1': 111,# NUM_1
	'2': 112,# NUM_2
	'3': 113,# NUM_3
	'4': 114,# NUM_4
	'5': 115,# NUM_5
	'6': 116,# NUM_6
	'7': 117,# NUM_7
	'8': 118,# NUM_8
	'9': 119,# NUM_9
	'$': 120,# DOLLAR
# Command.
	'##': 20, # EXTCMD
	'#?': 21, # EXTLIST
	'#a': 22, '\x01': 22, # ADJUST
	'#A': 23, # ANNOTATE
	'#x': 25, '\x18': 25, # ATTRIBUTES
	'#c': 29, '\x03': 29, # CHAT
	'#C': 31, # CONDUCT
	'#d': 32, # DIP
	'#e': 37, '\x05': 37, # ENHANCE
	'#f': 41, '\x06': 41, # FORCE
	'#i': 46, '\x09': 46, # INVOKE
	'#j': 47, '\x0A': 47, # JUMP
	'#k': 48, '\x04': 48, # KICK
	'#l': 52, '\x0C': 52, # LOOT
	'#m': 53, # MONSTER (use monster skill)
	'#o': 56, # OFFER/sacrifice
	'#O': 59, # OVERVIEW
	'#p': 62, # PRAY
	'#q': 65, # QUIT
	'#R': 70, # RIDE
	'#r': 71, # RUB
	'#s': 86, # SIT
	'\x14': 90, # TELEPORT
	'#t': 92, # TIP
	'#T': 94, # TURN undead
	'#u': 96, # UNTRAP
	'#V': 97, # VERSION
	'#w': 103,# WIPE
# MORE
	'|r': 68, # REDRAW
	'|a': 72, '|b': 73, # RUSH ?, RUSH2 ?
}

def getAction(_ord:bool=False)->int:
	from getch import Getch as getch
	key = getch().decode()
	if _ord: return ord(key)
	if key == '#' or key == '|':
		key += getch().decode()
	elif key == '\x1c': # Ctrl+\
		print('Terminate')
		return -1
	elif key == '^':
		from t0 import InteractiveEXEC
		InteractiveEXEC()
		return getAction(_ord)
	try:
		action = KEYMAP[key]
	except:
		print("unknown operation:", key)
		raise Exception(key)
	return action

def step(action:int):
	from nle_win import step as STEP
	obs, r, done = STEP(action)
	print(
		'%03d %5.2f'% (action, r),
		obs.misc[0], obs.misc[1], obs.misc[2]
	)
	return obs, r, done

if __name__ == '__main__':
	from nle_win import connect, disconnect, getobs
	from t0 import EXEC
	connect()
	print('start')
	EXEC('env = NLEnv(character="@", savedir=None)')
	EXEC('env.reset()')
	obs = getobs()
	done = False
	need_redraw = True
	while not done:
		if need_redraw:
			EXEC('env.env.render()')
			need_redraw = False
# make decision
		try:
			if obs.misc[0] and not obs.misc[1]: # input a line while don't require a <space> key
				key = input()
				if len(key):
					try:
						obs, r, done = step_primitive(key.encode())
						print(
							'%03d %5.2f'% (action, r),
							obs.misc[0], obs.misc[1], obs.misc[2]
						)
					except:
						print('quit')
						done = True
				else:
					obs, r, done = step(KEYMAP['\r'])
				need_redraw = True
			elif obs.misc[1]:
				key = chr(getAction(True))
				try:
					obs, done = step_primitive(key.encode())
					r = 0
					print(
						'%03d %5.2f'% (action, r),
						obs.misc[0], obs.misc[1], obs.misc[2]
					)
				except:
					print('quit')
					done = True
				need_redraw = True
				obs.misc[1]=0
			else:
				action = getAction()
				if action == -1:
					done = True
					raise Exception(action)
				obs, r, done = step(action)
				need_redraw = True
		except:
			pass
# take action
	print('done')
	EXEC('env.close()')
	disconnect()

'''
0 CompassDirection.N
1 CompassDirection.E
2 CompassDirection.S
3 CompassDirection.W
4 CompassDirection.NE
5 CompassDirection.SE
6 CompassDirection.SW
7 CompassDirection.NW
8 CompassDirectionLonger.N
9 CompassDirectionLonger.E
10 CompassDirectionLonger.S
11 CompassDirectionLonger.W
12 CompassDirectionLonger.NE
13 CompassDirectionLonger.SE
14 CompassDirectionLonger.SW
15 CompassDirectionLonger.NW
16 MiscDirection.UP
17 MiscDirection.DOWN
18 MiscDirection.WAIT
19 MiscAction.MORE
20 Command.EXTCMD
21 Command.EXTLIST
22 Command.ADJUST
23 Command.ANNOTATE
24 Command.APPLY
25 Command.ATTRIBUTES
26 Command.AUTOPICKUP
27 Command.CALL
28 Command.CAST
29 Command.CHAT
30 Command.CLOSE
31 Command.CONDUCT
32 Command.DIP
33 Command.DROP
34 Command.DROPTYPE
35 Command.EAT
36 Command.ENGRAVE
37 Command.ENHANCE
38 Command.ESC
39 Command.FIGHT
40 Command.FIRE
41 Command.FORCE
42 Command.GLANCE
43 Command.HISTORY
44 Command.INVENTORY
45 Command.INVENTTYPE
46 Command.INVOKE
47 Command.JUMP
48 Command.KICK
49 Command.KNOWN
50 Command.KNOWNCLASS
51 Command.LOOK
52 Command.LOOT
53 Command.MONSTER
54 Command.MOVE
55 Command.MOVEFAR
56 Command.OFFER
57 Command.OPEN
58 Command.OPTIONS
59 Command.OVERVIEW
60 Command.PAY
61 Command.PICKUP
62 Command.PRAY
63 Command.PUTON
64 Command.QUAFF
65 Command.QUIT
66 Command.QUIVER
67 Command.READ
68 Command.REDRAW
69 Command.REMOVE
70 Command.RIDE
71 Command.RUB
72 Command.RUSH
73 Command.RUSH2
74 Command.SAVE
75 Command.SEARCH
76 Command.SEEALL
77 Command.SEEAMULET
78 Command.SEEARMOR
79 Command.SEEGOLD
80 Command.SEERINGS
81 Command.SEESPELLS
82 Command.SEETOOLS
83 Command.SEETRAP
84 Command.SEEWEAPON
85 Command.SHELL
86 Command.SIT
87 Command.SWAP
88 Command.TAKEOFF
89 Command.TAKEOFFALL
90 Command.TELEPORT
91 Command.THROW
92 Command.TIP
93 Command.TRAVEL
94 Command.TURN
95 Command.TWOWEAPON
96 Command.UNTRAP
97 Command.VERSION
98 Command.VERSIONSHORT
99 Command.WEAR
100 Command.WHATDOES
101 Command.WHATIS
102 Command.WIELD
103 Command.WIPE
104 Command.ZAP
105 TextCharacters.PLUS
106 TextCharacters.MINUS
107 TextCharacters.SPACE
108 TextCharacters.APOS
109 TextCharacters.QUOTE
110 TextCharacters.NUM_0
111 TextCharacters.NUM_1
112 TextCharacters.NUM_2
113 TextCharacters.NUM_3
114 TextCharacters.NUM_4
115 TextCharacters.NUM_5
116 TextCharacters.NUM_6
117 TextCharacters.NUM_7
118 TextCharacters.NUM_8
119 TextCharacters.NUM_9
120 TextCharacters.DOLLAR
'''
