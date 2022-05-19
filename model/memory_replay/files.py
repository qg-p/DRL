def try_to_create_file(filename:str):
	import os
	if os.path.isfile(filename):
		filesize = os.path.getsize(filename)
		notEmptyFile = filesize != 0
		BKMG = ''
		if filesize>1024:
			filesize /= 1024
			BKMG = 'K'
		if filesize>1024:
			filesize /= 1024
			BKMG = 'M'
		if filesize>1024:
			filesize /= 1024
			BKMG = 'G'
		filesize = '%.1f %sB'%(filesize, BKMG)
		action = input('"{}" ({}) exists. Cover? [{}] '.format(filename, filesize, 'y/N' if notEmptyFile else 'Y/n'))
		if not len(action): action = 'n' if notEmptyFile else 'y' # default action
		yes = action[0] in 'yY' if notEmptyFile else action[0] not in 'nN' # do not deny
		if yes: print('"{}" will be covered.'.format(filename))
		else: return False
	else:
		pass # open(filename, 'wb').close() # can create
	return True

def iter_tmpfile(
	filename:str, past_filename:str=None,
	*, force_write:bool=False, do_not_cover:bool=False
):
	'''
	filename:
		新临时文件名，要保存到的文件名
	past_filename:
		旧临时文件名，要删除（移动）的文件名，None 表示创建新临时文件
	force_write: 如果 filename 已存在，
		True: 移除原来的 filename，移动旧文件到新文件，保存参数
		False: 选 Y 和 True 相同；选 N 不移除任何文件，也不保存参数
	do_not_cover: （被 force_write 覆盖）如果 filename 已存在
		True: 跳过选项选 N
		False: 控制台输入，阻塞
	'''
	import os
	# 新文件是否已存在
	if os.path.exists(filename):
		print('{} exists,'.format(filename), end=' ')
		if not force_write:
			if do_not_cover:
				action='n'
			else:
				action = input('cover? [Y/n]')
			if not len(action) and action[0] in 'nN':
				if past_filename is not None:
					print('"{}" is preserved.'.format(past_filename), end=' ')
				print('skip.')
				return False
		print('Cover.')

	if past_filename is None:
		last_filename = filename
	else: last_filename = past_filename
	# 旧文件存在且不是新文件，移动
	if os.path.exists(last_filename) and not (os.path.exists(filename) and os.path.samefile(filename, last_filename)):
		if os.path.exists(filename):
			os.remove(filename)
		os.rename(last_filename, filename)
	# 否则是同一文件，past_filename 为 None 或确实是一个文件，不重命名
	return True

from typing import List
def logfilexz_save_float(filename:str, floats:List[float]):
	'''直接存储 double (8B)，xzip 格式'''
	from dataset.xzfile import xz_file
	logfile = xz_file(filename, WR_ONLY=True)
	from ctypes import c_double
	double = c_double()
	for Float in floats:
		double.value=Float
		logfile.append(bytes(double))
	logfile.close()

def logfilexz_load_float(log_file_xz:str):
	from dataset.xzfile import xz_file
	file = xz_file(log_file_xz, RD_ONLY=True)
	from ctypes import c_double, sizeof
	floats = file.read()
	floats = (c_double*(len(floats)//sizeof(c_double))).from_buffer_copy(floats)
	floats = [*floats]
	return floats


def logfilexz_save_int(filename:str, ints:List[int]):
	'''直接存储 int (4B)，xzip 格式'''
	from dataset.xzfile import xz_file
	logfile = xz_file(filename, WR_ONLY=True)
	from ctypes import c_int
	long = c_int()
	for Int in ints:
		long.value=Int
		logfile.append(bytes(long))
	logfile.close()

def logfilexz_load_int(log_file_xz:str):
	from dataset.xzfile import xz_file
	file = xz_file(log_file_xz, RD_ONLY=True)
	from ctypes import c_int, sizeof
	longs = file.read()
	longs = (c_int*(len(longs)//sizeof(c_int))).from_buffer_copy(longs)
	longs = [*longs]
	return longs

def format_time():
	from time import strftime
	return strftime('%Y-%m%d-%H%M%S')
