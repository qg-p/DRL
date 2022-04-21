#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include "include/classes.h"
#include "include/wsl.so.h"

//FILE*fp[2];
int fd[2];
char Filename[2][256]={"pipea", "pipeb"};
char Filetype[2][2]={"w", "r"};
int fdtype[2]={O_WRONLY | O_BINARY, O_RDONLY | O_BINARY};
static void closepipe(bool verbose, int i) {
	if (fd[i]) {
		close(fd[i]);
	//	fp[i]=0;
		fd[i]=0;
	}
	else if (verbose)
		printf("%s(%d), %s:\t\'%s\' is already closed\n", __FILE__, __LINE__, __func__, Filename[i]);
}
void openPipe(char*filename, int i) {
	closepipe(false, i);
	if (filename[0])
		strcpy(Filename[i], filename);
	if (0 > (fd[i]=open(Filename[i], fdtype[i]))) {
		fd[i]=0;
		printf("%s(%d), %s:\tFailed to open %s file: \'%s\'\n", __FILE__, __LINE__, __func__, Filetype[i], Filename[i]);
		throw errNo::cannot_open;
	}
}
void closePipe(int i) {closepipe(true, i);}
static void openpipe(int i) {char _=0; openPipe(&_, i);}

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

errNo send(const varframe F) {
	const Frame&f=*F;
	unsigned char buffer[sizeof(f.head)+f.head.l], *i=buffer;
	int size=0;
	i+=size; memcpy(i, &f.head, size=sizeof(f.head));
	i+=size; memcpy(i, f.ptr, size=f.head.l);
	if (1!=fwriteSafe(buffer, sizeof(buffer), 1)) return errNo::cannot_write; // fwrite once
//	if (fflush(fp[1])) return errNo::warn;
//	closepipe(false, 1);
//	closepipe(false, 0);
	return errNo::success;
}

//	cannot_read: cannot read enough bytes from r pipe
//	too_long: bytes from r pipe longer than f.ptr_len is lost
errNo recv(varframe F) {
	Frame&f=*F;
	errNo retv=errNo::success;
//	if (fflush((fp[0]))) retv=errNo::warn;
	if (1!=freadSafe(&f.head, sizeof(f.head), 1))
		return errNo::cannot_read;
	unsigned size=f.head.l;
	if (f.head.l>f.ptr_len) {
		retv=errNo::too_long;
		size=f.ptr_len;
	} // fread twice if success else thrice
	if (1!=freadSafe(f.ptr, size, 1) && size) retv=errNo::cannot_read;
	if (size<f.head.l) {
		byte lost[f.head.l-size];
		if (1!=freadSafe(lost, sizeof(lost), 1)) retv=errNo::cannot_read;
	}
//	closepipe(false, 1);
//	closepipe(false, 0);
	return retv;
}

static struct Daemon {
	~Daemon() {
		for (auto&_:fd) {closepipe(false, &_-fd);}
	}
} DAEMON;
