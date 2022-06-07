if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/..'))
	del os, sys

from model.memory_replay.dataset.files import logfilexz_load_float
from matplotlib import pyplot as plt

file=input('float.xz file: ')
float_list = logfilexz_load_float(file)
plt.plot(float_list, marker='.', markersize=2, linewidth=0)
plt.show()