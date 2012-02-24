CC = g++
LD = g++
RM = rm -rf

CXXFLAGS = -O3 -funroll-all-loops
ROOTINCLUDEFLAGS = $(shell root-config --cflags) 

LDFLAGS =
ROOTLIBFLAGS = $(shell root-config --glibs)

SOURCES = $(wildcard *.cpp)

EXEC1 = PET_Analyse
EXEC2 = VoxelPhantom_Analyse

.PHONY: all clean $(EXEC1) $(EXEC2)

all: $(EXEC1) $(EXEC2)
	@echo done.

$(EXEC1): PET_Analyse.o
	@echo linking $@...
	@$(CC) -o $@ $^ $(LDFLAGS) $(ROOTLIBFLAGS)

$(EXEC2): VoxelPhantom_Analyse.o
	@echo linking $@...
	@$(CC) -o $@ $^ $(LDFLAGS) $(ROOTLIBFLAGS)

PET_Analyse.o: PET_Analyse.cpp
	@echo compiling $(notdir $<)...
	@$(CC) -o $@ -c $^ $(CXXFLAGS) $(ROOTINCLUDEFLAGS)

VoxelPhantom_Analyse.o: VoxelPhantom_Analyse.cpp
	@echo compiling $(notdir $<)...
	@$(CC) -o $@ -c $^ $(CXXFLAGS) $(ROOTINCLUDEFLAGS)

clean:
	@echo cleaning...
	@$(RM) *.o $(EXEC1) $(EXEC2)
