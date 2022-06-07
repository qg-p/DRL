if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

def main(batch_size:int, N:int):
	from time import perf_counter
	from nle_win.batch_nle.client import connect, disconnect, batch
	connect()
	env = batch(batch_size, 'character="@", savedir=None')
	t = perf_counter()
	env.reset()
	actions = [0]*batch_size
	for _ in range(N):
		env.step(actions)
	t = perf_counter() - t
	disconnect()
	return t
if __name__ == '__main__':
	batch_size = 64
	N = 100
	print(f'batch_size = {(batch_size)}, N_iter = {(N)}')
	t = main(batch_size, N)
	print(f'{(t)} s')
	print('%d/%.3f = %.6f iter/s' % (N*batch_size, t, N*batch_size/t))