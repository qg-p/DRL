if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
import nle_win as nle
# 三个 actions 序列，重要的是长度和 actions_normal 的内容
# 修改后需要到 exec_action 检查
actions_ynq = [ord(ch) for ch in [
	'\x1b', 'y', 'n', '*',
]]

actions_inv = [0, ord('$'), ord('#')]+[*range(ord('a'), ord('z')+1)]+[*range(ord('A'), ord('Z')+1)]

actions_normal = [ord(ch) for ch in [
	'k', 'l', 'h','j', 'u', 'n', 'b', 'y', # compass actions
	'<', '>', 's',
#	'c', # close
	'\x04', # kick
	',', # pick up
	# 后面的行动后接一个 * 将 misc 由 [1, 0, 0] 转为 [0, 0, 1]。
	'a', 'e', 'r', 'q', # apply, eat, read, quaff
	't', # throw
	'W', 'A', # wear, take off
	'w', # wield
	'\r', # enter
]]
actions_list = [actions_ynq, actions_inv, actions_normal]

def action_set_no(misc_6:list):
	if misc_6[3] and misc_6[4] and misc_6[5]:
		if misc_6[0]: return 0
		else: return 2 # y/n question repeats. minor bug
	elif misc_6[5] and not misc_6[3]: return 2
	elif misc_6[0] and misc_6[3]: return 1
	elif misc_6[0] or misc_6[3] or misc_6[2]: return 0
	else: return 2



def _select_action_human_input(no_action_set:int, obs:nle.basic.obs.observation):
	from getch import Getch
	print('>>> ', end='')
	action = Getch().decode()
	print(action)
	try:
		if no_action_set==1:
			from ctypes import string_at, c_uint8
			inv_letter = (c_uint8*len(obs.inv_letters))()
			for i, letter in enumerate(obs.inv_letters): inv_letter[i] = letter
			inv = string_at(inv_letter)
			action = inv.index(action)
		else:
			if no_action_set==2:
				action = {
					'k': 0, 'l': 1, 'j': 2, 'h': 3, 'u': 4, 'n': 5, 'b': 6, 'y': 7,
					'<': 16, '>': 17, '.': 75, 's': 75,
					'c': 30, '\x04': 48, ',': 61, # close, kick, pick up
				}[action]
			elif no_action_set==0:
				action = {
					' ': None, '\x1b': None, 'y': True, 'n': False,
				}[action]
			actions:list = actions_list[no_action_set]
			action = actions.index(action)
	except:
		action = 0
	return action
