from model.memory_replay.dataset.files import logfilexz_load_float
from matplotlib import pyplot as plt

file=input('float.xz file: ')
float_list = logfilexz_load_float(file)
plt.plot(float_list, marker='.', markersize=2, linewidth=0)
plt.show()