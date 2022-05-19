# from DRQN import DRQN

# from time import asctime
# def save_temp_parameters(model:DRQN, tag:str, filename_dataset:str=None, path:str=None):
# 	if path is None:
# 		import os
# 		path = os.path.normpath(os.path.dirname(os.path.abspath(__file__))+'/dat')
# 	if filename_dataset is None:
# 		filename_dataset = 'None'
# 	else:
# 		filename_dataset = os.path.split(filename_dataset)[1]
# 	filename = path+'/DRQN tag=[{}] dataset=[{}] time=[{}].pt'.format(tag, filename_dataset, asctime())
# 	model.save(filename)

# class logflie:
# 	'''
# 	记录 loss 的文件。
# 	注释以 0xffffffff 开始，以 0xffffffff 结束，8 字节填充。
# 	'''
# 	def __init__(self, filename:str):
# 		self.file = open(filename, 'ab+')
# 		self.comment_sign = b'\xFF'*4
# 	def append_comment(self, comment:str):
# 		comment = comment.encode()
# 	def append_double(self, loss:float):
# 		pass
# 	def read(self, n:int):
# 		pass
# 	def jump_to_comment(self):
# 		pass
# 	def close(self):
# 		pass

