from enum import IntEnum
class glyph_type(IntEnum):
	monster=0 # can attack
	food=1 # usually edible
	weapon=2 # left hand / right hand
	armor=3 # body armor
	ring=4 # left hand & right hand
	amulet=5 # neck
	tool=6 # applyable
	potion=7 # drinkable
	scroll=8 # read to zap
	spellbook=9 # read to learn spell
	wand=10 # zap
	stone=11 # gem, glass, grey stone, rock
	boulder=12 # boulder
	wall=13 # wall, rock, etc, unpassable
	room=14 # lit/dark:room/path, etc
	door=15 # passable in one direction if is opened
	stair=16 # stair, ladder, magic trap, vibrating square
	liquid=17 # water, lava, etc
	trap=18 # except magical trap & vibrating square
	explosion=19
	zap_beam=20
	no_item=21
	misc=22
	def max():
		return 22
def _get_descr():
	n_mon = 381 # 381 types of creature
	class Descr:
		def __init__(self, start:int, n:int, descr:str, type:glyph_type):
			self.start=start
			self.end=start+n
			self.descr=descr
			self.type=type
		def __call__(self, glyph:int):
			if self.start > glyph or self.end <= glyph:
				raise Exception()
			return self.descr+' #%d'%(glyph-self.start), self.type
		def __repr__(self) -> str:
			return '[%d, %d), %s, %s'%(self.start, self.end, self.descr, str(self.type))
	index = ( # modify here
		(n_mon  , 'monster'          , glyph_type.monster),
		(n_mon  , 'tamed monster'    , glyph_type.monster),
		(1      , 'invisible monster', glyph_type.monster),
		(n_mon  , 'detected monster' , glyph_type.monster),
		(n_mon  , 'monster corpse'   , glyph_type.food),
		(n_mon  , 'ridden monster'   , glyph_type.monster),
		(1      , 'illegal objects: strange object', glyph_type.monster),
# item (including boulder, venom and iron chain)
		(70     , 'weapon'    , glyph_type.weapon),
		(79     , 'armor'     , glyph_type.armor),
		(28     , 'ring'      , glyph_type.ring),
		(11     , 'amulet'    , glyph_type.amulet),
		(50     , 'tool'      , glyph_type.tool),
		(33     , 'food'      , glyph_type.food),
		(26     , 'potion'    , glyph_type.potion),
		(42     , 'scroll'    , glyph_type.scroll),
		(43     , 'spellbook' , glyph_type.spellbook),
		(27     , 'wand'      , glyph_type.wand),
		(1      , 'coin'      , glyph_type.tool),
		(22     , 'gem'       , glyph_type.stone),
		(9      , 'glass'     , glyph_type.stone),
		(4      , 'grey stone', glyph_type.tool),
		(1      , 'rock'      , glyph_type.stone),
		(1      , 'boulder'   , glyph_type.boulder),
		(1      , 'statue'    , glyph_type.room),
		(1      , 'iron ball' , glyph_type.tool),
		(1      , 'iron chain', glyph_type.tool),
		(2      , 'venom'     , glyph_type.misc),
# structure / terrain other than trap
		(1      , 'dark part of a room', glyph_type.wall), # 2359
		(11     , 'wall'               , glyph_type.wall),
		(1      , 'doorway'            , glyph_type.room),
		(2      , 'open door'          , glyph_type.door),
		(2      , 'closed door'        , glyph_type.door),
		(1      , 'iron bar'           , glyph_type.wall),
		(1      , 'tree'               , glyph_type.wall),
		(1      , 'floor of a room'    , glyph_type.room),
		(1      , 'dark part of a room', glyph_type.room), # 2379
		(2      , 'corridor'           , glyph_type.room), # normal, lit
		(4      , 'stair & ladder'     , glyph_type.stair), # su, sd, lu, ld
		(1      , 'altar'              , glyph_type.room),
		(1      , 'grave'              , glyph_type.room),
		(1      , 'throne'             , glyph_type.room),
		(2      , 'sink & fountain'    , glyph_type.room),
		(1      , 'water'              , glyph_type.liquid), # 2391
		(1      , 'ice'                , glyph_type.room),
		(1      , 'lava'               , glyph_type.liquid),
		(2      , 'lowered drawbridge' , glyph_type.door), # castle
		(2      , 'raised drawbridge'  , glyph_type.door), # castle
		(2      , 'air & cloud'        , glyph_type.room),
		(1      , 'water'              , glyph_type.liquid), # 2400
# trap
		(16     , 'trap'            , glyph_type.trap),
		(1      , 'migic portal'    , glyph_type.stair),
		(5      , 'trap'            , glyph_type.trap),
		(1      , 'vibrating square', glyph_type.stair),
# misc
		(12     , 'None'          , glyph_type.misc),
		(1      , 'poison cloud'  , glyph_type.room),
		(1      , 'valid position', glyph_type.room), # ?
		(8      , 'None'          , glyph_type.misc),
# damage area
		(63     , 'explosion', glyph_type.monster), # 3*3 square * 7 expl type
		(32     , 'beam'     , glyph_type.monster), # 4 direction * 8 beam type
# misc
		(n_mon*8, 'digesting monster'  , glyph_type.monster), # surrounded by gullet
		(6      , 'warning'            , glyph_type.monster),
		(n_mon  , 'monster statue'     , glyph_type.tool),
		(1      , 'empty'              , glyph_type.no_item),
	)
	descr:list[Descr] = [None]*len(index)
	start = 0
	for i, (n, s, t) in enumerate(index):
		descr[i]=Descr(start, n, s, t)
		start += n
	return descr
_descr = _get_descr()
def match_descr(glyph:int): # 区间二叉搜索
	# 空格 (2359) 最多，其次是 dark corridor (2380)、room (lit, dark) 和 wall
	# for i, descr in enumerate(_descr): # naive search
	# 	try:
	# 		desc = descr(glyph)
	# 		return i, descr, *desc
	# 	except:
	# 		pass
	l, r = 0, len(_descr)-1
	i = (l+r+1) // 2
	while True: # bi-search
		if glyph >= _descr[i].end:
			l = i
			i = (i+r+1) // 2
		elif glyph < _descr[i].start:
			r = i
			i = (i+l-1) // 2
		else:
			return i
def glyph_translate(glyph:int):
	i = match_descr(glyph)
	desc = _descr[i](glyph)
	return i, _descr[i], *desc
class terrain_type(IntEnum):
	item=6 # might be anything
	trap=5 # usually cause damage
	room=4 # passable from all 8 directions
	door=3 # passable in only 2 directions
	boulder=2 # can be eliminated using spell, tools, etc, can be pushed if is not obstructed
	monster=1
	wall=0 # cannot pass through
_gt = [terrain_type(0)]*(glyph_type.max()+1)
_gt[glyph_type.wall]=terrain_type.wall
_gt[glyph_type.monster]=terrain_type.monster
_gt[glyph_type.boulder]=terrain_type.boulder
_gt[glyph_type.food]=_gt[glyph_type.weapon]=_gt[glyph_type.armor]=_gt[glyph_type.ring]=_gt[glyph_type.amulet]=_gt[glyph_type.tool]=_gt[glyph_type.potion]=_gt[glyph_type.scroll]=_gt[glyph_type.spellbook]=_gt[glyph_type.wand]=_gt[glyph_type.stone]=_gt[glyph_type.misc]=_gt[glyph_type.no_item]=terrain_type.item
_gt[glyph_type.room]=_gt[glyph_type.stair]=terrain_type.room
_gt[glyph_type.door]=terrain_type.door
_gt[glyph_type.trap]=_gt[glyph_type.liquid]=_gt[glyph_type.explosion]=_gt[glyph_type.zap_beam]=terrain_type.trap

def _translate_glyph(glyph:int):
	no = match_descr(glyph)
	offset = glyph-_descr[no].start
	g_type = _descr[no].type
	t_type = _gt[g_type.value]
	return no, offset, g_type, t_type

from ctypes import c_short
def _get_glyph_array():
	glyph_list = [_translate_glyph(glyph) for glyph in range(_descr[len(_descr)-1].end)]
	_glyph_array = (c_short*4*_descr[len(_descr)-1].end)()
	for i, src in enumerate(glyph_list):
		_glyph_array[i] = src
	return _glyph_array
_glyph_array = _get_glyph_array()
def translate_glyph(glyph:int)->c_short*4: # 原有实现太低效了。CUDA 负载很低，但是 CPU 满载。
	return _glyph_array[glyph]

if __name__ == '__main__':
	import os, sys
	p=os.path.normpath(os.path.dirname(__file__)+'/../..')
	sys.path.append(p)
	del os, sys
import nle_win as nle
def translate_glyphs(obs:nle.basic.obs.observation):
	from ctypes import c_uint8
	import numpy as np
	glyphs_21_79 = obs.glyphs
	glyphs = np.array(glyphs_21_79)
	h, w = len(glyphs_21_79), len(glyphs_21_79[0])
	a = np.array((c_uint8*w*h*4)())
	for row, row_a in zip(glyphs, zip(a[0], a[1], a[2], a[3])):
		for i, glyph in enumerate(row):
			row_a[0][i], row_a[1][i], row_a[2][i], row_a[3][i] = translate_glyph(glyph)
	return a

def translate_inv(obs:nle.basic.obs.observation):
	inv_strs_55 = obs.inv_strs
	inv_glyphs_55 = obs.inv_glyphs
	inv_oclasses_55 = obs.inv_oclasses
	inv_size = len(inv_strs_55)
	from ctypes import string_at
	class BUC_status(IntEnum):
		unknown=0
		uncursed=1
		cursed=2
		blessed=3
	inv_strs = [string_at(inv_strs_55[i]).decode() for i in range(inv_size)]
	translation = [[0, 0 ,BUC_status(0), 0, 0, glyph_type(0), 0]] * inv_size
	for i, (strs, glyph, oclass) in enumerate(zip(inv_strs, inv_glyphs_55, inv_oclasses_55)):
		if len(strs):
			s = strs.split(' ')
			count = int(s[0]) if s[0] not in ['a', 'an', 'the'] else 1
		else:
			count = 0
		try:
			BUC = {
				'uncursed': BUC_status.uncursed,
				'cursed'  : BUC_status.cursed,
				'unholy'  : BUC_status.cursed,
				'blessed' : BUC_status.blessed,
				'holy'    : BUC_status.blessed,
			}[s[1]]
		except KeyError:
			BUC = BUC_status.unknown

		no = match_descr(glyph)
		offset = glyph-_descr[no].start
		g_type = _descr[no].type

		translation[i] = [
			int(oclass),
			count,
			BUC,
			no, offset, g_type,
		]
	return translation
def translate_messages_misc(obs:nle.basic.obs.observation):
	message_256 = obs.message
	misc_3 = obs.misc
	from ctypes import string_at
	message = string_at(message_256).decode()
	misc = [int(i) for i in misc_3]
	# message_has_more = '--More--' in message
	tty_message = bytes(obs.tty_chars[0]) # chatbox is also message
	tty_message_q_mark   = b'?' in tty_message
	tty_message_type   = b'type' in tty_message # 'What type ... take off?', 'What do ... take off?'
	tty_message_take_off = b'take off' in tty_message
	tty_message_pick_up  = b'Pick up what' in tty_message
	tty_message_You_read = b'You read: "' in tty_message
	# 不对 Put in what? 反应
	use_inv_tty_message = (tty_message_q_mark) and (not tty_message_You_read) and ((tty_message_take_off and not tty_message_type) and (not tty_message_pick_up))
	translation = [
		int(misc[0] or use_inv_tty_message), # whether an item is required
		misc[1], # 0 or 1 # use enter to skip # message_has_more or misc[1]
		misc[2], # chatbox
		int('? [' in message or use_inv_tty_message), # and ']' in message), # such as 'Really quit? [yn] (n)', 'What do you want to wield? [- a or *?]'
		int('? [yn' in message or '? [rl' in message), # y/n/q question, wear ring question
		int('?' in message),
	]
	return translation

if __name__ == '__main__':
	def __main__():
		import time
		t = time.process_time()
		for glyph in range(5977): glyph_translate(glyph)
		t = time.process_time() - t
		for i, g in enumerate(_descr): # 测试是否为划分
			u, v = glyph_translate(g.start)[2], glyph_translate(g.end-1)[2]
			if (u.split('#')[0]!=v.split('#')[0]):
				print('error')
				print(i, u, v)
		print(i+1)
		print('time', t, 's')
		# for i in _descr: print(i)

	__main__()
	o=nle.basic.obs.observation()
	print(translate_glyphs(o))