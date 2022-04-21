#include <cstdio>
#include <cstring>
#include <unistd.h>

FILE*fp;
int fd;
char Filename[256]="pipe";
void _closePipe(bool);
extern "C" void closePipe() {_closePipe(true);}
static struct Stat {
	static int n_call_close;
	Stat() {}
	~Stat() {
		if (n_call_close>10)
			printf("%d fclose\n", n_call_close);
		_closePipe(false);
	}
} stat;
int Stat::n_call_close=0;
void _closePipe(bool verbose) {
	if (!fp) {
		if (verbose)
			printf("%s(%d)|%s:\t\'%s\' is already closed\n", __FILE__, __LINE__, __func__, Filename);
		return;
	}
	Stat::n_call_close++;
	fclose(fp);
	fp=0;
	fd=0;
}
extern "C" void openPipe(char*filename) {
	_closePipe(false);
	if (filename[0])
		strcpy(Filename, filename);
	if (!(fp = fopen(Filename, "w"))) {
		printf("%s(%d)|%s:\tFailed to open file: \'%s\'\n", __FILE__, __LINE__, __func__, Filename);
		throw 0;
	}
	fd = fileno(fp);
}
void _openPipe() {char _=0; openPipe(&_);}
int fwrite_nobuf(void*ptr, size_t size, int n, FILE*fp) { // fp is not used
	int N=write(fd, ptr, size*n);
	return size!=0 ? N>0 ? N/size : 0 : 1;
}
// send much, receive little
extern "C" {
	#include "./observation.h"
	observation buffer;
	void send_obs(
		short*glyphs,             // 21 * 79
		char*chars,               // 21 * 79
		char*colors,              // 21 * 79
		char*specials,            // 21 * 79
		int64_t*blstats,          // 26
		char*message,             // 256
		short*inv_glyphs,         // 55
		char*inv_strs,            // 55 * 80
		char*inv_letters,         // 55
		char*inv_oclasses,        // 55
	//	char*screen_descriptions, // 21 * 79 * 80
		char*tty_chars,           // 24 * 80
		char*tty_colors,          // 24 * 80
		char*tty_cursor,          // 2
		int*misc                  // 3
	) {	// sum: 149941 Bytes per observation, actually: 149944 Bytes(4-bytes alignment)
		if (!fp) _openPipe();
		memcpy(&buffer.glyphs, glyphs, sizeof(buffer.glyphs));
		memcpy(&buffer.chars, chars, sizeof(buffer.chars));
		memcpy(&buffer.colors, colors, sizeof(buffer.colors));
		memcpy(&buffer.specials, specials, sizeof(buffer.specials));
		memcpy(&buffer.blstats, blstats, sizeof(buffer.blstats));
		memcpy(&buffer.message, message, sizeof(buffer.message));
		memcpy(&buffer.inv_glyphs, inv_glyphs, sizeof(buffer.inv_glyphs));
		memcpy(&buffer.inv_strs, inv_strs, sizeof(buffer.inv_strs));
		memcpy(&buffer.inv_letters, inv_letters, sizeof(buffer.inv_letters));
		memcpy(&buffer.inv_oclasses, inv_oclasses, sizeof(buffer.inv_oclasses));
	//	memcpy(&buffer.screen_descriptions, screen_descriptions, sizeof(buffer.screen_descriptions));
		memcpy(&buffer.tty_chars, tty_chars, sizeof(buffer.tty_chars));
		memcpy(&buffer.tty_colors, tty_colors, sizeof(buffer.tty_colors));
		memcpy(&buffer.tty_cursor, tty_cursor, sizeof(buffer.tty_cursor));
		memcpy(&buffer.tty_cursor, misc, sizeof(buffer.misc));
		if (fwrite_nobuf(&buffer, sizeof(buffer), 1, fp)!=1) {
			printf("%s(%d)|%s:\tFailed to write\n", __FILE__, __LINE__, __func__);
			throw 0;
		}
	//	_close(false);
	}
}
