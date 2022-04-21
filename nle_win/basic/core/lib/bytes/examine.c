#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

FILE*FP[2];

void*print_pipe(void*filename) {
	FILE*fp=fopen(filename, "r");
	FP[0]=fp;
	printf("read start\n");
	int ch=0;
	while (ch!=-1) {
		fflush(fp);
		ch = getc(fp);
		fprintf(stderr, "%02X ", ch);
	}
	printf("read end\n");
	FP[0]=0;
	return 0;
}

void*write_zero(void*filename) {
	FILE*fp=fopen(filename, "w");
	FP[1]=fp;
	printf("write begin\n");
	while (getchar()!=~0) {
		fputc(0, fp);
		fflush(fp);
	}
	printf("write end\n");
	FP[1]=0;
	return 0;
}

void*refresh(void*_) {
	while (!(FP[1]&&FP[0]))
		usleep(50000);
	while (FP[1]||FP[0]) {
		if (FP[0]) fflush(FP[0]);
		if (FP[1]) fflush(FP[1]);
		usleep(50000);
	}
	return 0;
}

int main(int argc, char*argv[]) {
	if (argc!=2) {
		printf("usage : %s [ 0 | 1 ]\n", argv[0]);
		return 0;
	}
	int n=atoi(argv[1]);
	FILE*fp[2];
	const char*Filename[2];
	if (n)
		Filename[0]="pipea", Filename[1]="pipeb";
	else
		Filename[0]="pipeb", Filename[1]="pipea";
	pthread_t pid1, pid2;
	pthread_create(&pid1, 0, write_zero, (void*)Filename[0]);
	pthread_create(&pid2, 0, print_pipe, (void*)Filename[1]);
	refresh(0);
	pthread_join(pid1, 0);
	pthread_join(pid2, 0);
	return 0;
}
