from ctypes import Structure
class batch_unit(Structure):
	import ctypes
	from .. import basic
	_fields_ = (
		('obs', basic.obs.observation),
		('reward', ctypes.c_float),
		('done', ctypes.c_uint32),
	)
	class template:
		import ctypes
		from .. import basic
		obs    = basic.obs.observation()
		reward = ctypes.c_float()
		done   = ctypes.c_uint32()
	def __init__(self, *args, **kwargs) -> None:
		self.obs    = batch_unit.template.obs
		self.reward = batch_unit.template.reward
		self.done   = batch_unit.template.done
		super().__init__(*args, **kwargs)

class batch_frame():
	def __init__(self, batch_size:int, package_offset=0, package_length=0, package_ID=0):
		from ..basic.common import Type
		from ..basic.core.bytes import classes
		from ctypes import cast, sizeof, POINTER

		sizeof_batch = batch_size*sizeof(batch_unit)
		ptr = (classes.c_ubyte*sizeof_batch)()
		ptr_len = sizeof(ptr)
		type = Type['observation_batch']

		self.batch_size = batch_size

		self.f = classes.Frame(
			ptr=cast(ptr, classes.c_ubyte_p),
			ptr_len=ptr_len,
			head=classes.Frame_header(
				l=sizeof_batch,
				type=type,
				head=classes.Pkg_header(
					pkg_offset = package_offset,
					pkg_len = package_length,
					pkg_ID = package_ID
				)
			)
		)

		self.batch = cast(ptr, POINTER(batch_unit))

	def __getitem__(self, index:int):
		return self.batch[index]

	def send(self)->int:
		from ctypes import pointer
		from ..basic.core.bytes import send
		return send(pointer(self.f))

	def recv(self)->int:
		from ctypes import pointer
		from ..basic.core.bytes import recv
		return recv(pointer(self.f))
	
	def type(self)->int:
		return self.f.head.head.type

	def __len__(self):
		return int(self.f.head.l)

#	def __sizeof__(self):
#		return self.f.__sizeof__() + self.f.ptr_len

