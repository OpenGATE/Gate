INCLUDE = -I../includes/

ROOTLIBS     := $(shell root-config --libs)

CC = gcc
CFLAGS = $(INCLUDE) -D_FILE_OFFSET_BITS=64
#-pg

ifeq ($(BUILD),debug)
CFLAGS += -g -Wall
endif

LDFLAGS = $(INCLUDE) -L../lib/ -lLMF
LDFLAGS += $(ROOTLIBS)
# -lefence
SYSTEM := $(shell uname)
ifeq ($(SYSTEM),SunOS)
	LDFLAGS += -lxnet
endif

COBJ = $(wildcard *.c)
OBJ = $(patsubst %.c,%.o,$(COBJ))

all : EXE_01 EXE_02 EXE_04 EXE_05 EXE_06 EXE_07 EXE_08 END_OF_LINK

%%.o:%%.c $(CFLAGS)

EXE_01 : exempleMain_01.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 1 : EXE_01 ok..."	

EXE_02 : exempleMain_02.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 2 : EXE_02 ok..."	

EXE_04 : exempleMain_04.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 4 : EXE_04 ok..."	

EXE_05 : exempleMain_05.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 5 : EXE_05 ok..."	

EXE_06 : exempleMain_06.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 6 : EXE_06 ok..."	

EXE_07 : exempleMain_07.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 7 : EXE_07 ok..."	

EXE_08 : exempleMain_08.o
	gcc -o $@ $^ $(LDFLAGS) 
	@echo "example 8 : EXE_08 ok..."	


END_OF_LINK :
	@echo ""
	@echo ""
	@echo "Congratulations : compilation and linkage over."
	@echo "Type EXE_01 to execute first example"
	@echo ""
	@echo ""

clean :		 
	-rm *~
	-rm core
	-rm *#
	-rm *.o
	-rm EXE*
	-clear
	-ls
