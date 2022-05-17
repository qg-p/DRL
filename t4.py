# test batch server & client
term='!'
def Exec():
	while True:
		cmd=input('> ')
		if cmd==term: break
		try:
			exec(cmd)
		except:
			from traceback import print_exc
			print(print_exc())
if __name__ == '__main__':
	from nle_win.batch_nle import connect, disconnect, EXEC, batch, terminate
	from model.misc import action_set_no
	from model.glyphs import translate_messages_misc
	connect()
	env = batch(1, 'character="Val-Hum-Fem-Law", savedir=None, penalty_step=-0.01')
	print('start')
	line = env.reset()[0]
	obs, reward, done = line.obs, line.reward, line.done
	need_redraw = True
	while not done:
		if need_redraw:
			EXEC('env.render(0)')
			need_redraw = False
# make decision
		from getch import Getch
		action = Getch()
		if action == term.encode():
			Exec()
			continue
		line = env.step([action[0]])[0]
		obs, reward, done = line.obs, line.reward, line.done
		print('blstats: {}'.format([*obs.blstats]))
		print('inv_letters: {}'.format([*obs.inv_letters]))
		print('misc: {} {} {}'.format([*obs.misc], translate_messages_misc(obs), action_set_no(translate_messages_misc(obs))))
		print('reward: %g'%(reward))
		need_redraw = True
# take action
	print('done')
	EXEC('env.render(0)')
	EXEC('env.close()')
	disconnect()
