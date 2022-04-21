import nle
from nle import nethack as nh


def main():
	for glyph in range(nh.MAX_GLYPH):
		name = glyph2name(glyph)
		if name is not None:
			print(glyph, name)


TMB = ("top", "middle", "bottom")
LCR = ("left", "center", "right")

ZAPS = ("missile", "fire", "frost", "sleep", "death", "lightning", "gas", "acid")
ZDESCS = ("vbeam", "hbeam", "lslant", "rslant")

EXPLS = ("dark", "noxious", "muddy", "wet", "magical", "fiery", "frosty")


def glyph2name(glyph):
	if glyph >= nh.MAX_GLYPH:
		return None
	if glyph >= nh.GLYPH_STATUE_OFF:
		off = glyph - nh.GLYPH_STATUE_OFF
		return nh.permonst(off).mname + " statue"
	if glyph >= nh.GLYPH_WARNING_OFF:
		off = glyph - nh.GLYPH_WARNING_OFF
		return "warning %d" % off
	if glyph >= nh.GLYPH_SWALLOW_OFF:
		off = glyph - nh.GLYPH_SWALLOW_OFF
		piece = off & 0x7
		piece += int(piece > 3)
		mon = nh.permonst((off & ~0x7) >> 3)
		return "swallowed: gullet of %s, %s %s" % (
			mon.mname,
			TMB[piece // 3],
			LCR[piece % 3],
		)
	if glyph >= nh.GLYPH_ZAP_OFF:
		off = glyph - nh.GLYPH_ZAP_OFF
		piece = off & 0x3
		zapnum = (off & ~0x3) >> 2
		return "%s zap, %s" % (ZAPS[zapnum], ZDESCS[piece])
	if glyph >= nh.GLYPH_EXPLODE_OFF:
		off = glyph - nh.GLYPH_EXPLODE_OFF
		explnum = off // nh.MAXEXPCHARS
		piece = off % nh.MAXEXPCHARS
		return "%s explosion, %s %s" % (
			EXPLS[explnum],
			TMB[piece // 3],
			LCR[piece % 3],
		)
	if glyph >= nh.GLYPH_CMAP_OFF:
		off = glyph - nh.GLYPH_CMAP_OFF
		return nh.symdef.from_idx(off).explanation
	if glyph >= nh.GLYPH_OBJ_OFF:
		off = glyph - nh.GLYPH_OBJ_OFF
		obj = nh.objclass(off)
		return "%s: %s" % (
			nh.class_sym.from_oc_class(obj.oc_class).name,
			nh.OBJ_NAME(obj),
		)
	if glyph >= nh.GLYPH_RIDDEN_OFF:
		off = glyph - nh.GLYPH_RIDDEN_OFF
		return "ridden %s" % nh.permonst(off).mname
	if glyph >= nh.GLYPH_BODY_OFF:
		off = glyph - nh.GLYPH_BODY_OFF
		return "%s corpse" % nh.permonst(off).mname
	if glyph >= nh.GLYPH_DETECT_OFF:
		off = glyph - nh.GLYPH_DETECT_OFF
		return "detected %s" % nh.permonst(off).mname
	if glyph >= nh.GLYPH_INVIS_OFF:
		return "invisible monster"
	if glyph >= nh.GLYPH_PET_OFF:
		off = glyph - nh.GLYPH_PET_OFF
		return "tame %s" % nh.permonst(off).mname
	if glyph >= nh.GLYPH_MON_OFF:
		off = glyph - nh.GLYPH_MON_OFF
		return nh.permonst(off).mname
	return None


if __name__ == "__main__":
	main()
