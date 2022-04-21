def test2():
	print('test srv step rate')
	from time import perf_counter as sec
	from nle_win import Exec, step, getobs
	from random import random as rand
	Exec("env = NLEnv('NetHackChallenge-v0', savedir=None)")
	Exec('env.reset()')
	obs = getobs()
	cnt = 0
	totalR = 0.
	done = False
	MAX = 121
	ls = [0] * MAX
	t0 = sec()
	while not done:
		if obs.misc[1]:
			action = 38 # ESC
		elif obs.misc[0]:
			action = 19 # MORE, enter, '\r'
		else:
			action = MAX
			while action == MAX or action == 65:
				action = round(rand()*MAX)
			ls[action] += 1
		obs, reward, done = step(action)
		cnt += 1
		totalR += reward
	t1 = sec() - t0
	Exec('print(env.info)')
	Exec('env.close()')
	print(ls)
	print('total reward: %f' % (totalR))
	print('cnt: %d, t: %.4fs, %.2f op/s' % (cnt, t1, cnt/t1 if t1!=0 else -1))

if __name__ == '__main__':
	from nle_win import disconnect, connect
	connect()
	test2()
	disconnect()
