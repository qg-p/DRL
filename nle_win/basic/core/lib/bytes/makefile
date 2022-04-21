#!/usr/bin/make
#----------- CONFIG -----------
targetA=wslmain
targetB=winmain.exe
libA=wsl.so
libB=win.dll

src=src
include=$(src)/include
bin=bin
obj=obj

ifeq ($(task),)
task=build
#\
task:=run
#\
task:=debug
endif

#------------ Envs ------------
argsA=
argsB=
gdbinstr=--ex "b main"

#------------ FLAG ------------
ccA=g++
ccB=gcc.exe
cf=-m64 -pipe -Wall -fPIC
lf=

ifeq ($(task), debug)
cf+=-O0 -g
else
cf+=-O
lf:=-s $(lf)
endif

#------------ Task ------------
all: $(task)
build: $(bin)/$(targetA) $(bin)/$(targetB) examine
run: build pipea pipeb
	$(bin)/$(targetB) $(argsB) &
	$(bin)/$(targetA) $(argsA)
debug: build pipea pipeb
	gdb -q -tui $(gdbinstr) --ex "r $(argsA) > /dev/null" $(bin)/$(targetA)
.PHONY: all build run debug

#---- Compile & Clean cmds ----
#pipe:
#	mkfifo $@

$(bin)/$(targetA): $(obj)/$(targetA).o $(bin)/$(libA)
	$(ccA) $^ $(cf) -o $@ $(lf)
$(obj)/$(targetA).o: $(src)/$(targetA).c* $(include)/classes.h $(include)/wsl.so.h
	$(ccA) $< $(cf) -c -o $@
$(bin)/$(targetB): $(obj)/$(targetB).o $(bin)/$(libB)
	$(ccB) $^ $(cf) -o $@ $(lf)
	chmod +xxx $@
$(obj)/$(targetB).o: $(src)/$(targetB).c* $(include)/classes.h $(include)/win.dll.h
	$(ccB) $< $(cf) -c -o $@
clean1:
	rm -f $(obj)/$(targetA).o $(obj)/$(targetB).o

$(bin)/$(libA): $(obj)/$(libA).o
	$(ccA) $< $(cf) --share -o $@
$(obj)/$(libA).o: $(src)/$(libA).c* $(include)/classes.h
	$(ccA) $< $(cf) -c -o $@
$(bin)/$(libB): $(obj)/$(libB).o
	$(ccB) $< $(cf) --share -o $@
$(obj)/$(libB).o: $(src)/$(libB).c* $(include)/classes.h
	$(ccB) $< $(cf) -c -o $@
clean2:
	rm -f $(obj)/$(libA).o $(obj)/$(libB).o
	rm -f $(bin)/$(libA) $(bin)/$(libB)

clean: clean1 clean2
	rm -f $(bin)/$(targetA) $(bin)/$(targetB)
.PHONY: clean clean1 clean2

examine: examine.c
	gcc -pthread $< -O -s -o $@
