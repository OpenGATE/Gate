
#=========================================================
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

#=========================================================
PROJECT(Gate)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules ${CMAKE_MODULE_PATH})

#=========================================================
# Option to use visualisation feature of G4
OPTION(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
IF (WITH_GEANT4_UIVIS)
      FIND_PACKAGE(Geant4 REQUIRED ui_all vis_all)
ELSE()
      FIND_PACKAGE(Geant4 REQUIRED)
ENDIF()


#=========================================================
# Build variables for G4 Version

#MESSAGE("G4 Version ${Geant4_VERSION}")
string(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\1" G4VERSION_MAJOR ${Geant4_VERSION})
string(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\2" G4VERSION_MINOR ${Geant4_VERSION})
string(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\3" G4VERSION_PATCH ${Geant4_VERSION})
#MESSAGE(${G4VERSION_MAJOR})
#MESSAGE(${G4VERSION_MINOR})
#MESSAGE(${G4VERSION_PATCH})

#=========================================================
# Option for Root use

OPTION(GATE_USE_ROOT "Gate use ROOT (Recommanded)" ON)
OPTION(GATE_USE_ANALYSIS_USE_GENERAL "Gate use  G4ANALYSIS_USE_GENERAL (Recommanded)" ON)
OPTION(GATE_USE_OWN_CLHEP "Gate does not use the standard CLHEP of GEANT4" ON)
OPTION(GATE_USE_FILE "Gate use G4ANALYSIS_USE_FILE (Recommanded)" ON)

IF (GATE_USE_ROOT)
   FIND_PACKAGE(ROOT REQUIRED)
   INCLUDE_DIRECTORIES(${ROOT_INCLUDE_DIR})
   SET(LINK_DIRECTORIES ${ROOT_LIBRARY_DIR})
   # MESSAGE("Found Root in ${ROOT_INCLUDE_DIR} ${ROOT_LIBRARY_DIR}")
ENDIF(GATE_USE_ROOT)

IF (GATE_USE_OWN_CLHEP)
   FIND_PACKAGE(CLHEP REQUIRED)
   INCLUDE_DIRECTORIES(${CLHEP_INCLUDE_DIR})
   SET(LINK_DIRECTORIES ${CLHEP_LIBRARIES})
   #MESSAGE("Found Clhep in ${CLHEP_INCLUDE_DIR} ${CLHEP_LIBRARY}")
ENDIF(GATE_USE_OWN_CLHEP)

# G4ANALYSIS_USE_GENERAL
# G4ANALYSIS_USE_FILE
# GATE_USE_OPTICAL
# GATE_USE_LMF
# GATE_USE_ECAT7
# G4ANALYSIS_USE_ROOT_PLOTTER

#=========================================================
# Create configuration file
CONFIGURE_FILE(GateConfiguration.h.in ${PROJECT_BINARY_DIR}/GateConfiguration.h)
include_directories(${PROJECT_BINARY_DIR})

#=========================================================
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
include(${Geant4_USE_FILE})
include_directories(
	${PROJECT_SOURCE_DIR}/source/arf/include
	${PROJECT_SOURCE_DIR}/source/geometry/include
	${PROJECT_SOURCE_DIR}/source/physics/include
	${PROJECT_SOURCE_DIR}/source/digits_hits/include
	${PROJECT_SOURCE_DIR}/source/general/include)

#=========================================================
# Locate sources and headers for this project
file(GLOB sources
	${PROJECT_SOURCE_DIR}/source/arf/src/*.cc
	${PROJECT_SOURCE_DIR}/source/geometry/src/*.cc
	${PROJECT_SOURCE_DIR}/source/digits_hits/src/*.cc
	${PROJECT_SOURCE_DIR}/source/physics/src/*.cc
	${PROJECT_SOURCE_DIR}/source/general/src/*.cc)
file(GLOB headers
	${PROJECT_SOURCE_DIR}/source/arf/include/*.hh
	${PROJECT_SOURCE_DIR}/source/geometry/include/*.hh
	${PROJECT_SOURCE_DIR}/source/physics/include/*.hh
	${PROJECT_SOURCE_DIR}/source/digits_hits/include/*.hh
	${PROJECT_SOURCE_DIR}/source/general/include/*.hh)

#=========================================================
# Add the executable, and link it to the Geant4 libraries
add_executable(Gate Gate.cc ${sources} ${headers})
target_link_libraries(Gate ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${CLHEP_LIBRARIES})

#=========================================================
install(TARGETS Gate DESTINATION bin)
