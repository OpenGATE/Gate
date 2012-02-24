gatename := fGate
name := gjm
EXE     = bin/$(G4SYSTEM)/$(name)
SRCDIRS = . src 
INCDIRS = . include $(ROOTSYS)/include
LIBS    = `$(ROOTSYS)/bin/root-config --libs`

CC      = g++
LFLAGS  = -s
CFLAGS  = -Wall -O2 -fPIC

SRC     = $(foreach dir,$(SRCDIRS),$(wildcard $(dir)/*.cc))
OBJ     = $(addprefix tmp/, $(SRC:.cc=.o))
DEP     = $(addprefix tmp/,$(SRC:.cc=.d))
INCLUDE = $(addprefix -I, $(INCDIRS) )

.PHONY: clean dirs

all: $(EXE) copy
$(EXE): dirs $(OBJ)
	@echo Linking $(name) ...
	@$(CC) -o $(EXE) $(OBJ) $(LFLAGS) $(LIBS)
	@echo ... Done!

tmp/%.o: %.cc
	@echo  Creating object file for $*.cc ...
	@$(CC) -Wp,-MMD,tmp/$*.dd $(CFLAGS) $(INCLUDE) -c $< -o $@
	@sed -e '1s/^/$(subst /,\/,$(dir $@))/' tmp/$*.dd > $*_tmp; mv $*_tmp tmp/$*.d
	@rm -f tmp/$*.dd


%.hh: ;
		

clean:
	@echo Cleaning up ...
	@-rm -rf tmp
	@-rm -rf bin

copy:
#ifdef GATEHOME
	@-mkdir -p $(GATEHOME)/bin; mkdir -p $(GATEHOME)/bin/$(G4SYSTEM)
	@-cp bin/$(G4SYSTEM)/$(name) $(GATEHOME)/bin/$(G4SYSTEM)/
#endif

dirs:
	@-if [ ! -e tmp ]; then mkdir tmp; fi;
	@-if [ ! -e bin ]; then mkdir bin; mkdir bin/$(G4SYSTEM); fi;
	@-$(foreach DIR,$(SRCDIRS), if [ ! -e tmp/$(DIR) ]; then mkdir tmp/$(DIR); fi; )

-include $(DEP)
