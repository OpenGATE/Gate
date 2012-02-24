# --------------------------------------------------------------
name := Gate
G4TARGET := $(name)
G4EXLIB := true

G4LIB_BUILD_SHARED=yes

if $(G4WORKDIR)=='.'
  G4WORKDIR=../..

G4TMP=$(G4WORKDIR)/tmp
G4BIN=$(G4WORKDIR)/bin
 	

CPPFLAGS += -I../arf/include \
            -I../geometry/include \
            -I../physics/include 	\
            -I../digits_hits/include \
            -I../general/include \

	

include $(G4INSTALL)/config/binmake.gmk

include $(G4WORKDIR)/GateCommon.gmk


