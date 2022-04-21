#ifndef wsl_so_h
#define wsl_so_h

enum errNo:int {
	other=-1, success=0, warn=1,
	cannot_open, cannot_read, cannot_write,
	too_long,
};

extern "C" {
void openPipe(char*filename, int file_index);
void closePipe(int file_index);
int freadSafe(const void*ptr, int size, int n);
int fwriteSafe(const void*ptr, int size, int n);
}
#endif // wsl_so_h
