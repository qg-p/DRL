if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

if __name__ == '__main__':
	from test_batch_env_rate import main
	# from model.memory_replay.dataset.files import logfilexz_save_float
	from matplotlib import pyplot as plt
	max_batch_size = 128
	N = 100
	bss = [*range(1, max_batch_size+1)]
	ts = [main(bs, N) for bs in bss]
	print(ts)
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.set_ylabel('time (s)')
	ax2.set_ylabel('rate (iter/s)')
	ax1.set_xlabel('batch size')
	lines =  ax1.plot(bss, ts, label='time (N = %d)'%(N), color = 'C0')
	lines += ax2.plot(bss, [N*bs/t for bs, t in zip(bss, ts)], label='rate', color = 'C1')
	plt.legend(lines, [l.get_label() for l in lines])
	plt.show()