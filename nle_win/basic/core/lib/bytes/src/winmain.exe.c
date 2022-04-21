#include "include/classes.h"
#include "include/win.dll.h"
#include <stdio.h>

byte BUFFER[163840];
int BUFFERlen() {return sizeof(BUFFER);}
byte* getBUFFER() {return BUFFER;}
int main(int argc, char*argv[], char*env[]) {
	openPipe("", 1);
	openPipe("", 0);

	Frame frcv={.ptr=BUFFER, .ptr_len=BUFFERlen()};
	int i=10;
	while (i) {
		errNo no=recv(&frcv);
		if (no==success) break;
		else printf("win: recv: %d %d\n", no, --i);
	}
	printf("win: %s\n", frcv.ptr);
	frcv.ptr[0]=0;
//	closePipe(1);

	char bfr[]="windows*1";
	Frame fsnd={
		.ptr=(void*)bfr,// .ptr_len=sizeof(bfr),
		.head={
			.l=sizeof(bfr)-1,
			.head={
				.type=bytes,
				.pkg_ID=new_ID(), .pkg_len=sizeof(bfr)-1, .pkg_offset=0
			}
		}
	};
	errNo no=send(&fsnd);
	if (no!=success) printf("win: send: %d\n", no);
	//closePipe(0);

	no=send(&frcv);
	if (no!=success) printf("win: send: %d\n", no);
	//closePipe(0);

	i=5;
	while (i) {
		errNo no=recv(&frcv);
		if (no==success) break;
		else printf("win: recv: %d %d\n", no, --i);
	}
	printf("win: %s\n", frcv.ptr);
	frcv.ptr[0]=0;
//	closePipe(1);

	closePipe(0);
	return 0;
}
