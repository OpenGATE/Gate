CXX := g++
LD := g++

CP := cp
RM := rm -rf

MAINSOURCES := $(wildcard *.cc)
SOURCES := $(wildcard ./src/*.cc)
MAINOBJECTS := $(patsubst %.cc, tmp/%.o, $(notdir $(MAINSOURCES)))
OBJECTS := $(patsubst %.cc, tmp/%.o, $(notdir $(SOURCES)))

vpath %.hh ./include
vpath %.cc ./src

CXXFLAGS :=
INCLUDE := -I./include `geant4-config --cflags` `root-config --cflags`
LDFLAGS := `geant4-config --libs` `root-config --glibs`

TARGET := gjm

.PHONY: all clean directories cleanall install uninstall

all: directories $(TARGET)
	@echo Done

directories:
	@if test ! -f tmp;\
	then mkdir -p tmp;\
	fi

$(TARGET): $(MAINOBJECTS) $(OBJECTS)
	@echo Linking...
	@$(LD) -o $@ $^ $(INCLUDE) $(LDFLAGS)

tmp/gjm.o: gjm.cc GateMergeManager.hh
	@echo Compiling $(notdir $<)...
	@$(CXX) -o $@ -c $< $(INCLUDE) $(CXXFLAGS)

tmp/GateMergeManager.o: GateMergeManager.cc GateMergeManager.hh
	@echo Compiling $(notdir $<)...
	@$(CXX) -o $@ -c $< $(INCLUDE) $(CXXFLAGS)

clean:
	@echo Cleaning...
	@$(RM) $(OBJECTS) $(TARGET) $(MAINOBJECTS)

cleanall: clean
	@$(RM) tmp

install:
	@echo Installing ...
	@$(CP) $(TARGET) /usr/local/bin

uninstall:
	@echo Uninstalling...
	@$(RM) /usr/local/bin/$(TARGET)
