if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/../../..'))
	del os, sys
# import nle_win as nle
from nle_win.batch_nle.client import connect, disconnect, batch, EXEC
from dataset.dataset import ARS_dataset, ARS_dataset_xz
from getch import Getch
from model.misc import actions_list, action_set_no as set_no
from model.glyphs import translate_messages_misc as t
from model import setting

def main(filename:str):
	connect()
	env_param = 'character="Val-Hum-Fem-Law", savedir=None, penalty_step={}'.format(setting.penalty_step)
	env = batch(1, env_param)
	d = ARS_dataset_xz(filename, env_param, False) # xz不支持边读边写
	action = 255
	done = True
	while True:
		# make decision
		if done:
			action = 255
			done = input('New game? [Y/n] ')
			if len(done) != 0 and done[0] in ['N', 'n']:
				break
			print('Start.')
			done = False
			no = None
		else:
			last_no = no
			no = set_no(t(state))
			if no == 1:
				actions = [c for c in state.inv_letters if c != 0]+[ord('\r')]
				# if not len(actions): actions = [0]
			else:
				actions = actions_list[no]
			# print (new) valid actions
			if last_no != no or no == 1:
				special_actions = {
					0:'<0>',
					0x1b:'<Esc>',
					ord('\r'):'<CR>',
					4:'<Ctrl-D>',
				}
				actions_print = ''
				for action in actions:
					if action in special_actions.keys():
						action = special_actions[action]
					else:
						action = chr(action)
					actions_print += action + ' '
				print(actions_print)
			# input action
			while True:
				action = Getch()[0]
				if no == 1 and actions[0] == 0:
					action = 0
				if action not in actions:
					print('Invalid action.')
				else: break
		# take action
		[unit] = env.step([action])
		EXEC('env.render(0)')
		state = unit.obs
		reward = unit.reward
		done = bool(unit.done)
		d.append_line(action, reward, state)
	d.append_line(action, 0., state)
	print('End.')
	disconnect()

if __name__ == '__main__':
	import os
	filename = os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/dat/test-Val-Hum-Fem-Law.ARS.dat.xz')
	main(filename)