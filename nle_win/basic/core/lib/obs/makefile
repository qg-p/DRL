all: build
build: lib.dll lib.so
.PHONY: all build

cflags=-pipe -O3 -fPIC -shared
lib.dll: src/dll.c src/observation.h
	gcc.exe $< $(cflags) -o $@
lib.so: src/so.cpp src/observation.h
	wsl.exe -- g++ $< $(cflags) -o $@

clean:
	wsl.exe -- rm -f lib.dll lib.so
.PHONY: clean