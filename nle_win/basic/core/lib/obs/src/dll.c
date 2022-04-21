#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
FILE*fp;
int fd;
char Filename[256]="pipe";
typedef unsigned char bool;
#define true  ((bool)1)
#define false ((bool)0)
void _closePipe(bool verbose) {
	if (!fp) {
		if (verbose)
			printf("%s(%d)|%s:\t\'%s\' is already closed\n", __FILE__, __LINE__, __func__, Filename);
		return;
	}
	fclose(fp);
	fp=0;
	fd=0;
}
void closePipe() {_closePipe(true);}
void openPipe(char*filename) { // open the file of last time until 'filename' is specified, default to "pipe"
	_closePipe(false);
	if (filename[0])
		strcpy(Filename, filename);
	if (!(fp = fopen(Filename, "rb"))) {
		printf("%s(%d)|%s:\tFailed to open file: \'%s\'\n", __FILE__, __LINE__, __func__, Filename);
		exit(0);
	}
	fd = fileno(fp);
}
void _openPipe() {char _=0; openPipe(&_);}

int fread_nobuf(void*ptr, size_t size, int n, FILE*fp) { // fp is not used
	int N=read(fd, ptr, n*size);
	return size!=0 ? N>0 ? N/size : 0 : 1;
}
// receive much, send little
#include "./observation.h"
int recv_obs(struct observation*target) {
	if (!fp) // not initialized
		_openPipe();
	if (fread_nobuf(target, sizeof(struct observation), 1, fp)!=1) { // passive close
		printf("%s(%d)|%s:\tBroken file \'%s\'\n", __FILE__, __LINE__, __func__, Filename);
		return 1;
	}
	return 0;
}
//void send_act(int action) {
//	;
//}
