def From(lib:str):
	try:
		From.cdll
	except:
		from ctypes import cdll
		From.cdll = cdll
		del cdll
		return From(lib)
	try:
		From.Library
	except:
		From.Library = {} # Library Cache
		return From(lib)
	if lib not in From.Library.keys():
		From.Library[lib] = From.cdll.LoadLibrary(lib)
	dll = From.Library[lib]
	def Import(Imitation):
		setattr(Imitation.Func, Imitation.func, getattr(dll, Imitation.func))
		del Imitation.Func, Imitation.func
		def imitation(*args, **kwargs):
			return Imitation(*args, **kwargs)
		return imitation
	return Import

def Import(func:str):
	def decorator(Func):
		def Imitation(*args, **kwargs):
			return Func(Func, *args, **kwargs)
		Imitation.func = func # 要加载的函数名
		Imitation.Func = Func # 传递内层函数给 From.Import
		return Imitation
	return decorator
