#include "include/classes.h"
#include "include/wsl.so.h"
#include <stdio.h>
#include <unistd.h>

byte BUFFER[163840];
int BUFFERlen() {return sizeof(BUFFER);}
byte* getBUFFER() {return BUFFER;}
int main(int argc, char*argv[], char*env[]) {
	openPipe((char*)"", 0);
	openPipe((char*)"", 1);

	char bfr[]="linux*3";
	Frame fsnd(
		bfr, sizeof(bfr), sizeof(bfr)-1,
		bytes,
		0, sizeof(bfr)-1, new_ID()
	);
	errNo no=send(&fsnd);
	if (no!=success) printf("WSL: send: %d\n", no);
//	closePipe(0);

	Frame frcv(BUFFER, BUFFERlen());
	int i=8;
	while (i) {
		no=recv(&frcv);
		if (no==success) break;
		else printf("WSL: recv: %d %d\r", no, --i);
	}
	printf("WSL: %s\n", frcv.ptr);
	frcv.ptr[0]=0;
//	closePipe(1);

	i=4;
	while (i) {
		no=recv(&frcv);
		if (no==success) break;
		else printf("WSL: recv: %d %d\r", no, --i);
	}
	printf("%s\n", frcv.ptr);
	frcv.ptr[0]=0;
//	closePipe(1);

	no=send(&frcv);
	if (no!=success) printf("WSL: send: %d\n", no);
	//closePipe(0);

	return 0;
}
