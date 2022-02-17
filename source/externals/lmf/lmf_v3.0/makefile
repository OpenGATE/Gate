INCLUDE = -I./includes/

ROOTCFLAGS   := $(shell root-config --cflags)

ARCH := $(shell uname -p)

LIB = ./lib/libLMF.a 
#LIB = ./lib/libLMF.so

CFLAGS = $(INCLUDE) -D_FILE_OFFSET_BITS=64 -fPIC
LDFLAGS = $(INCLUDE)

ifeq ($(BUILD),debug)
CFLAGS += -g -Wall
endif

ifeq ($(ARCH),x86_64)
CFLAGS += -D_64
endif

CSRC = $(wildcard src/*.c)
CCSRC = $(wildcard src/*.cc)
SRC = $(CSRC)
SRC += $ $(CCSRC)
OBJ = $(patsubst src/%.c,obj/%.o,$(CSRC))
OBJ += $(patsubst src/%.cc,obj/%.o,$(CCSRC))

all : $(LIB) TAGS END_OF_COMPILE

obj/%.o : src/%.c
	gcc $(CFLAGS) -c -o $@ $<

obj/outputRootMgr.o : src/outputRootMgr.cc
	gcc $(CFLAGS) $(ROOTCFLAGS) -c -o $@ $<

obj/%.o : src/%.cc
	gcc $(CFLAGS) -c -o $@ $<



$(LIB) : $(OBJ)
	ar rc $@ $^ 
#	gcc -o $@ -shared $^ 


END_OF_COMPILE :
	@echo ""
	@echo ""
	@echo ""
	@echo "Congratulations : compilation and linkage of lib/$(LIB) over !!!"
	@echo "Please try make in ./exemples to compile our examples"
	@echo ""
	@echo ""


clean :		 
	-rm *~
	-rm core
	-rm *#
	-rm obj/*.o
	-rm $(LIB)
	-clear
	-ls

PATHLMFCOMMON = /opt/lmf
install : 
	-chmod -R 700 $(PATHLMFCOMMON)
	-rm -rf $(PATHLMFCOMMON)/$(LIB)
	-rm -rf $(PATHLMFCOMMON)/includes/*	
	-cp $(LIB) $(PATHLMFCOMMON)/$(LIB)
	-cp includes/* $(PATHLMFCOMMON)/includes/
	-chmod -R 555 $(PATHLMFCOMMON)

TAGS :
	etags src/*.{c,cc} includes/*.{h,hh}
