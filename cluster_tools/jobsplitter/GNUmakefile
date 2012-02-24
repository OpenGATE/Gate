gatename := fGate
# define your default platform
#DEFAULT_PLATFORM := openmosix
DEFAULT_PLATFORM := condor
#DEFAULT_PLATFORM := openPBS


CPPFLAGS += -DGC_DEFAULT_PLATFORM=\"$(DEFAULT_PLATFORM)\"

name := gjs
G4TARGET := $(name)
G4EXLIB := true
G4TMP=$(G4WORKDIR)/tmp
G4BIN=$(G4WORKDIR)/bin

ifndef G4INSTALL
  G4INSTALL = ../../..
endif

.PHONY: all
all: lib bin copy

include $(G4INSTALL)/config/architecture.gmk

ifdef G4VERSION4
  CPPFLAGS += -DG4VERSION4
endif

ifdef G4VERSION5_1
  CPPFLAGS += -DG4VERSION5_1
endif

ifdef G4VERSION5_2
  CPPFLAGS += -DG4VERSION5_2
endif

include $(G4INSTALL)/config/binmake.gmk

LDFLAGS += -L/usr/X11R6/lib -lXt
ifdef G4VIS_USE_OPENGLXM
LDFLAGS += -lXm
endif

CXXFLAGS := $(filter-out -ansi , $(CXXFLAGS))
CXXFLAGS := $(filter-out -pedantic , $(CXXFLAGS))

CXXFLAGS_WITHOUT_O := $(filter-out -O% , $(CXXFLAGS))
CXXFLAGS_WITHOUT_O := $(filter-out +O% , $(CXXFLAGS_WITHOUT_O))

CPPFLAGS := $(filter-out -I/usr/include , $(CPPFLAGS))

copy:
#ifdef GATEHOME
	@-mkdir -p $(GATEHOME)/bin; mkdir -p $(GATEHOME)/bin/$(G4SYSTEM)
	@-mkdir -p $(GATEHOME)/tmp; mkdir -p $(GATEHOME)/tmp/$(G4SYSTEM); mkdir -p $(GATEHOME)/tmp/$(G4SYSTEM)/$(gatename)
	@-cp $(G4WORKDIR)/bin/$(G4SYSTEM)/$(name) $(GATEHOME)/bin/$(G4SYSTEM)/
	@-cp $(G4WORKDIR)/tmp/$(G4SYSTEM)/$(name)/lib$(name)* $(GATEHOME)/tmp/$(G4SYSTEM)/$(gatename)/
#endif
