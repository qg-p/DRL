#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "include/classes.h"
#include "include/win.dll.h"

void Exit(int);
//FILE*fp[2];
int fd[2];
char Filename[2][256]={"pipeb", "pipea"};
char Filetype[2][2]={"w", "r"};
int fdtype[2]={O_WRONLY | O_BINARY, O_RDONLY | O_BINARY};
typedef unsigned char bool;
#define true  ((bool)1)
#define false ((bool)0)
void closepipe(bool verbose, int i) {
	if (fd[i]) {
		close(fd[i]);
		fd[i]=0;
	}
	else if (verbose)
		printf("%s(%d), %s:\t\'%s\' is already closed\n", __FILE__, __LINE__, __func__, Filename[i]);
}
void openPipe(char*filename, int i) {
	closepipe(false, i);
	if (filename[0])
		strcpy(Filename[i], filename);
	if (0 > (fd[i] = open(Filename[i], fdtype[i]))) {
		fd[i]=0;
		printf("%s(%d), %s:\tFailed to open %s file: \'%s\'\n", __FILE__, __LINE__, __func__, Filetype[i], Filename[i]);
		Exit(0);
	};
}
void closePipe(int i) {closepipe(true, i);}
void openpipe(int i) {char _=0; openPipe(&_, i);} // open the last file that is opened

int freadSafe(const void*ptr, int size, int n) {
	if (!fd[1]) openpipe(1);
	size*=n;
	return read(fd[1], (void*)ptr, size) == size ? n : 0;
}
int fwriteSafe(const void*ptr, int size, int n) {
	if (!fd[0]) openpipe(0);
	size*=n;
	return write(fd[0], ptr, size) == size ? n : 0;
}

errNo send(varframe F) {
	unsigned char buffer[sizeof((*F).head)+(*F).head.l], *i=buffer;
	int size=0;
	i+=size; memcpy(i, &(*F).head, size=sizeof((*F).head));
	i+=size; memcpy(i, (*F).ptr, size=(*F).head.l);
	if (1!=fwriteSafe(buffer, sizeof(buffer), 1)) return cannot_write; // fwrite once
//	closepipe(false, 1);
//	closepipe(false, 0);
	return success;
}

errNo recv(varframe F) {
//	if (!fp[1]) openPipe("", 1);
//	fflush(fp[1]);
//	(*F).head.l=getw(fp[1]); // fixed bug: exe blocks correctly with this line, prog goes on if and only if pipe is written
	if (1!=freadSafe(&(*F).head, sizeof((*F).head), 1))
		return cannot_read;
	errNo retv=success;
	unsigned size=(*F).head.l;
	if ((*F).head.l>(*F).ptr_len) {
		retv=too_long;
		size=(*F).ptr_len;
	}
	if (1!=freadSafe((*F).ptr, size, 1) && size) retv=cannot_read;
	if (size<(*F).head.l) {
		byte waste[(*F).head.l-size];
		if (1!=freadSafe(waste, sizeof(waste), 1)) retv=cannot_read;
	}
//	closepipe(false, 1);
//	closepipe(false, 0);
	return retv;
}

void Exit(int __status) {
	closepipe(false, 0);
	closepipe(false, 1);
	exit(__status);
}
