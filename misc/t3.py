if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

# primitive only

from nle_win.client import step_primitive

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
		from getch import Getch
		action = Getch()
		obs, r, done = step_primitive(action)
		print([*obs.blstats])
		print(r)
		need_redraw = True
# take action
	print('done')
	EXEC('env.close()')
	disconnect()
