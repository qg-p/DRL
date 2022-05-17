if __name__=='__main__':
	import os, sys
	sys.path.append(os.path.normpath(os.path.dirname(__file__)+'/..'))
	del os, sys
import nle_win as nle

from ctypes import Structure, c_int32, c_double
class ARS_Line(Structure):
	'''
	action:	int 32
	reward:	float 64
	state:	observation

	action == -1 -> state = env.reset(), done(last_state) = True
	'''
	template={
		'action': c_int32(),
		'reward': c_double(),
		'state' : nle.basic.obs.observation(),
	}
	_fields_=[(K, type(V)) for K, V in template.items()]
	def __init__(self, *args, **kwargs) -> None:
		self.action:c_int32 = self.template['action']
		self.reward:c_double = self.template['reward']
		self.state:nle.basic.obs.observation = self.template['state']
		super().__init__(*args, **kwargs)

class dataset():
	from typing import Type
	from ctypes import Structure
	from io import BufferedRandom
	def __init__(self, file:BufferedRandom, UnitType:Type[Structure]) -> None:
		self.file = file
		self.UnitType = UnitType

		self.pos = self.file.tell()
		self.file_size = self.file.seek(0, 2)
		self.pos = self.file.seek(self.pos, 0)
		# assert self.pos == offset

	def read(self, n_unit:int):
		from ctypes import sizeof #, POINTER, cast
		size = sizeof(self.UnitType)
		assert size*n_unit+self.pos<=self.file_size # 可读入若干数据
		buffer = self.file.read(size*n_unit) # bytes
		self.pos = self.file.tell()
		base = (self.UnitType*n_unit).from_buffer_copy(buffer) # 深拷贝
		l = [*base] # split，引用
		return l
	def append_lines(self, lines:list):
		self.file.seek(0, 2)
		for unit in lines:
			if not isinstance(unit, self.UnitType): break
			self.file.write(unit)
		self.file_size = self.file.tell()
		self.pos = self.file.seek(self.pos, 0)
		assert isinstance(unit, self.UnitType)

class ARS_dataset(dataset):
	def __init__(self, filename:str, metadata:str='', keep_exist_data:bool=True) -> None:
		'''
		filename: name of dataset file
		metadata: comment to write in the head of file if keep_exist_data is False or it has no data
		keep_exist_data: keep existing data. only read and append.
		'''
		file = open(filename, 'ab+')
		if file.tell()==0 or not keep_exist_data:
			# WR=True # whether clear data
			file = open(filename, 'wb+')
			metadata = metadata.__repr__()+'\n'
			file.write(metadata.encode())
			metadata = eval(metadata[:-1])
		else:
			file.seek(0, 0)
			metadata_new = eval(file.readline()[:-1].decode())
			if metadata and metadata != metadata_new:
				print('ignore: {}'.format(metadata))
			metadata = metadata_new
		self.metadata = metadata
		super().__init__(file, ARS_Line)
	def append_line(self, action:int, reward:float, obs:nle.basic.obs.observation):
		line = ARS_Line(action=action, reward=reward, state=obs)
		self.append_lines([line])
	from typing import List
	def read(self, n_unit:int)->List[ARS_Line]:
		return dataset.read(self, n_unit)
	def readall(self)->List[ARS_Line]:
		from ctypes import sizeof
		return self.read((self.file_size-self.pos)//sizeof(self.UnitType))

class ARS_dataset_xz():
	def __init__(self, filename:str, metadata:str='', keep_exist_data:bool=True) -> None:
		'''
		filename: name of dataset.xz file (lzma)
		metadata: comment to write in the head of file if keep_exist_data is False or it has no data
		keep_exist_data: keep existing data. only read and append.
		'''
		from .xzfile import xz_file
		self.file = xz_file(filename, RD_ONLY=keep_exist_data, WR_ONLY=not keep_exist_data)
		import os
		if os.path.isfile(filename) and keep_exist_data:
			metadata_new = eval(self.file.readline().decode())
			if metadata and metadata != metadata_new:
				print('ignore: {}'.format(metadata))
			metadata = metadata_new
		else:
			metadata = metadata.__repr__()+'\n'
			self.file.append(metadata.encode())
			metadata = eval(metadata)
		self.metadata = metadata

	from typing import List
	def read(self, n_unit:int)->List[ARS_Line]:
		if self.file.wrbuffer is not None:
			return [] # writeonly
		from ctypes import sizeof
		size = sizeof(ARS_Line)
		assert size*n_unit+self.file.pos<=self.file.file_size
		buffer = self.file.read(size*n_unit)
		base = (ARS_Line*n_unit).from_buffer_copy(buffer)
		return [*base]
	def readall(self):
		from ctypes import sizeof
		return self.read((self.file.file_size-self.file.pos)//sizeof(ARS_Line))
	def append_line(self, action:int, reward:float, obs:nle.basic.obs.observation):
		line = ARS_Line(action=action, reward=reward, state=obs)
		self.append_lines([line])
	def append_lines(self, lines:List[ARS_Line]):
		for unit in lines:
			self.file.append(bytes(unit))

def __main__():
	def test_rdwr():
		nle.connect()
		env_param = '(character="Val-Hum-Fem-Law", savedir=None, penalty_step=-0.01)'
		d = ARS_dataset('dataset/dat/0-Val-Hum-Fem-Law.ARS.dat', env_param, False)
		nle.Exec('env=NLEnv'+env_param)
		nle.Exec('env.reset()')
		obs = nle.getobs()
		d.append_line(-1, 0., obs)
		nle.disconnect()
		line = d.read(1)[0]
		print(line)
		from ctypes import string_at
		action, reward, message = line.action, line.reward, string_at(line.state.message).decode()
		print(action, reward, message)
		print(d.pos, d.file_size)
		assert all([d.pos == d.file_size, action==-1, reward==0, len(message)])
	test_rdwr()

if __name__=='__main__':
	__main__()