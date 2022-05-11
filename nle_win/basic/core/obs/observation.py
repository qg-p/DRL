from ctypes import Structure, c_int32, c_int64, c_short, c_uint8
class observation(Structure):
	_fields_ = [
		('glyphs'             , c_short * 79 * 21),
		('chars'              , c_uint8 * 79 * 21),
		('colors'             , c_uint8 * 79 * 21),
		('specials'           , c_uint8 * 79 * 21),
		('blstats'            , c_int64 * 26),
		('message'            , c_uint8 * 256),
		('inv_glyphs'         , c_short * 55),
		('inv_strs'           , c_uint8 * 80 * 55),
		('inv_letters'        , c_uint8 * 55),
		('inv_oclasses'       , c_uint8 * 55),
	#	('screen_descriptions', c_uint8 * 80 * 79 * 21),
		('tty_chars'          , c_uint8 * 80 * 24),
		('tty_colors'         , c_uint8 * 80 * 24),
		('tty_cursor'         , c_uint8 * 2),
		('misc'               , c_int32 * 3)
	]
	class template:
		glyphs       = (c_short * 79 * 21)()
		chars        = (c_uint8 * 79 * 21)()
		colors       = (c_uint8 * 79 * 21)()
		specials     = (c_uint8 * 79 * 21)()
		blstats      = (c_int64 * 26)()
		message      = (c_uint8 * 256)()
		inv_glyphs   = (c_short * 55)()
		inv_strs     = (c_uint8 * 80 * 55)()
		inv_letters  = (c_uint8 * 55)()
		inv_oclasses = (c_uint8 * 55)()
		tty_chars    = (c_uint8 * 80 * 24)()
		tty_colors   = (c_uint8 * 80 * 24)()
		tty_cursor   = (c_uint8 * 2)()
		misc         = (c_int32 * 3)()
	def __init__(self, *args, **kwargs) -> None:
		self.glyphs       = observation.template.glyphs      
		self.chars        = observation.template.chars
		self.colors       = observation.template.colors
		self.specials     = observation.template.specials
		self.blstats      = observation.template.blstats
		self.message      = observation.template.message
		self.inv_glyphs   = observation.template.inv_glyphs
		self.inv_strs     = observation.template.inv_strs
		self.inv_letters  = observation.template.inv_letters
		self.inv_oclasses = observation.template.inv_oclasses
		self.tty_chars    = observation.template.tty_chars
		self.tty_colors   = observation.template.tty_colors
		self.tty_cursor   = observation.template.tty_cursor
		self.misc         = observation.template.misc        
		super().__init__(*args, **kwargs)