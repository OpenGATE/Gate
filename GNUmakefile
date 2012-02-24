# --------------------------------------------------------------
name := Gate
G4TARGET := $(name)
G4EXLIB := true

SUBDIRS := source/arf source/geometry source/general source/digits_hits source/physics 

G4LIB_BUILD_SHARED=yes

G4TMP=$(G4WORKDIR)/tmp
G4BIN=$(G4WORKDIR)/bin

.PHONY: makesub all bin clean cleansub
all: makesub lib bin

include $(G4WORKDIR)/GateCommon.gmk
include $(G4INSTALL)/config/binmake.gmk

CPPFLAGS += -I./source/arf/include \
            -I./source/geometry/include \
            -I./source/physics/include 	\
            -I./source/digits_hits/include \
            -I./source/general/include \

makesub:
	@for dir in $(SUBDIRS); do ( \
		echo Entering $$dir ... ; \
		cd $$dir; \
		$(MAKE) obj; \
                echo $$dir finished );\
	done
