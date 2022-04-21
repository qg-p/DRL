#include <stdint.h>
//#pragma pack(1) // python Structure is always aligned
struct observation {
	int16_t glyphs              [21][79];
	int8_t  chars               [21][79];
	int8_t  colors              [21][79];
	int8_t  specials            [21][79];
	int64_t blstats             [26];
	int8_t  message             [256];
	int16_t inv_glyphs          [55];
	int8_t  inv_strs            [55][80];
	int8_t  inv_letters         [55];
	int8_t  inv_oclasses        [55];
//	int8_t  screen_descriptions [21][79][80];
	int8_t  tty_chars           [24][80];
	int8_t  tty_colors          [24][80];
	int8_t  tty_cursor          [2];
	int32_t	misc				[3];
};
//#pragma pack(0)
