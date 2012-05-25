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

CXXFLAGS := -DGC_DEFAULT_PLATFORM=\"condor\"
INCLUDE := -I./include `geant4-config --cflags`
LDFLAGS := `geant4-config --libs`

TARGET := gjs

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

tmp/gjs.o: gjs.cc GateSplitManager.hh
	@echo Compiling $(notdir $<)...
	@$(CXX) -o $@ -c $< $(INCLUDE) $(CXXFLAGS)

tmp/GateMacfileParser.o: GateMacfileParser.cc GateMacfileParser.hh
	@echo Compiling $(notdir $<)...
	@$(CXX) -o $@ -c $< $(INCLUDE) $(CXXFLAGS)

tmp/GateSplitManager.o: GateSplitManager.cc GateSplitManager.hh
	@echo Compiling $(notdir $<)...
	@$(CXX) -o $@ -c $< $(INCLUDE) $(CXXFLAGS)

tmp/GateToPlatform.o: GateToPlatform.cc GateToPlatform.hh
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
