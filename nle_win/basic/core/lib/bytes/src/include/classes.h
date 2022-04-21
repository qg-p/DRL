#ifndef classes_h
#define classes_h
#ifdef __cplusplus
extern "C" {
#endif

#ifdef _O_BINARY
#define O_BINARY _O_BINARY // 0x8000, MinGW-W64, <fcntl.h>
#else
#define O_BINARY 0
#endif

typedef unsigned char byte;

typedef enum Type {
	bytes=0, // send a vector
	command,
	observation, reward, done, info,
	action,
} Type;

#ifdef __cplusplus
enum errNo:int;
#else
typedef enum errNo errNo;
#endif

typedef struct Frame Frame;
#ifdef __cplusplus
typedef Frame*varframe; // Frame&
#else
typedef Frame*varframe;
#endif
errNo send(const varframe f); // send protocol to utilize
errNo recv(varframe f); // recv protocol to utilize

unsigned new_ID() {
	static unsigned ID;
	return ++ID;
}
typedef struct Package_head {
	Type type;
	unsigned
		pkg_offset,
		pkg_len, pkg_ID;
#ifdef __cplusplus
	Package_head(Type type, unsigned offset, unsigned package_len, unsigned package_ID)
	:	type(type), pkg_offset(offset), pkg_len(package_len), pkg_ID(package_ID)
	{}
	Package_head(){}
#endif
} Package_head;
typedef struct Frame_head { // first struct to fill from r pipe
	unsigned l; // Frame::ptr+l
	Package_head head;
#ifdef __cplusplus
	Frame_head
	(	unsigned length,
		Type type, unsigned offset, unsigned package_len, unsigned package_ID)
	:	l(length), head(type, offset, package_len, package_ID){}
	Frame_head(){}
#endif
} Frame_head;
struct Frame { // struct to store all info from r pipe / contains into to send to w pipe
	byte* ptr;
	unsigned ptr_len;
	Frame_head head;
#ifdef __cplusplus
	Frame
	(	void*pointer=0, unsigned buffer_length=0, unsigned length=0,
		Type type=bytes, unsigned package_offset=0,
		unsigned package_len=0, unsigned package_ID=0)
	:	ptr((byte*)pointer), ptr_len(buffer_length), head(length, type, package_offset, package_len, package_ID)
	{}
	Frame(){}
	byte*begin() {return ptr;}
	byte*end() {return ptr+head.l;}
	errNo send() {return ::send(this);}
	errNo recv() {return ::recv(this);}
#endif
};



#ifdef __cplusplus
}
#endif
#endif // classes_h
