class xz_file():
	class non_xz_Exception(Exception):
		def __init__(self, *args: object) -> None:
			super().__init__(*args)
	class non_file_Exception(Exception):
		def __init__(self, *args: object) -> None:
			super().__init__(*args)
	def __init__(self, filename:str, *, RD_ONLY:bool=None, WR_ONLY:bool=None) -> None:
		if RD_ONLY is None and WR_ONLY is None:
			RD_ONLY = os.path.isfile(filename)
		if RD_ONLY is None and WR_ONLY is not None:
			RD_ONLY = not WR_ONLY
		if WR_ONLY is None and RD_ONLY is not None:
			WR_ONLY = not RD_ONLY
		if WR_ONLY is not None and RD_ONLY is not None:
			assert bool(WR_ONLY) != bool(RD_ONLY)
		keep_exist_data = RD_ONLY

		import os
		if os.path.exists(filename) and not os.path.isfile(filename):
			raise xz_file.non_file_Exception(filename, 'is not a file')
		if os.path.splitext(filename)[1] not in ('.xz', '.xzip'):
			raise xz_file.non_xz_Exception(filename, 'is not a .xz file')

		import lzma
		self.lzma = lzma
		if os.path.isfile(filename) and keep_exist_data:
			file = self.lzma.open(filename)
			self.wrbuffer:bytes = None
		else:
			file = open(filename, 'wb')
			self.wrbuffer = bytes()
		self.file = file

		self.pos = self.file.tell()
		self.file_size = self.file.seek(0, 2)
		self.pos = self.file.seek(self.pos, 0)
	def append(self, readable:bytes):
		# assert self.wrbuffer is not None
		self.wrbuffer += readable
		self.file_size += len(readable)
		return self.file_size
	def read(self, size:int=None):
		assert self.wrbuffer is None
		buffer = self.file.read(size)
		self.pos += len(buffer)
		return buffer
	def readline(self, size:int=None):
		assert self.wrbuffer is None
		return self.file.readline(size)
	def close(self):
		if self.wrbuffer is not None:
			buffer = self.lzma.compress(self.wrbuffer)
			self.file.write(buffer)
		self.file.close()
		self.wrbuffer = None
	def seek(self, offset:int, whence:int):
		return self.file.seek(offset, whence)
	def tell(self):
		if self.wrbuffer is None:
			return self.file.tell()
		return len(self.wrbuffer)
	def __del__(self):
		self.close()
