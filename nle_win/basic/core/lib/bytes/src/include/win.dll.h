#ifndef win_dll_h
#define win_dll_h

enum errNo {
	other=-1, success=0, warn=1,
	cannot_open, cannot_read, cannot_write,
	too_long
};

void openPipe(char*filename, int file_index);
void closePipe(int file_index);
int freadSafe(const void*ptr, int size, int n);
int fwriteSafe(const void*ptr, int size, int n);
void Exit(int);

#endif // win_dll_h
