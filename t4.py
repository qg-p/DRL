# test batch server & client
if __name__ == '__main__':
	from nle_win.batch_nle import connect, disconnect, EXEC, batch, terminate
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
			line = env.step([action[0]])[0]
			obs, reward, done = line.obs, line.reward, line.done
			print([*obs.blstats])
			print(reward)
			need_redraw = True
# take action
	print('done')
	EXEC('env.close()')
	disconnect()
