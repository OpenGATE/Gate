#=========================================================
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

#=========================================================
PROJECT(Gate)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules ${CMAKE_MODULE_PATH})

#=========================================================
# Option to use visualisation feature of G4
OPTION(WITH_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
IF(WITH_GEANT4_UIVIS)
      FIND_PACKAGE(Geant4 REQUIRED ui_all vis_all)
ELSE()
      FIND_PACKAGE(Geant4 REQUIRED)
ENDIF()


#=========================================================
# Build variables for G4 Version

#MESSAGE("G4 Version ${Geant4_VERSION}")
STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\1" G4VERSION_MAJOR ${Geant4_VERSION})
STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\2" G4VERSION_MINOR ${Geant4_VERSION})
STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\3" G4VERSION_PATCH ${Geant4_VERSION})
#MESSAGE(${G4VERSION_MAJOR})
#MESSAGE(${G4VERSION_MINOR})
#MESSAGE(${G4VERSION_PATCH})

#=========================================================
# Option for Root use

OPTION(GATE_USE_ROOT "Gate use ROOT (Recommanded)" ON)
OPTION(GATE_USE_OPTICAL "Gate use optical photon" OFF)
OPTION(GATE_ANALYSIS_USE_GENERAL "Gate use  G4ANALYSIS_USE_GENERAL (Recommanded)" ON)
OPTION(GATE_ANALYSIS_USE_FILE "Gate use G4ANALYSIS_USE_FILE (Recommanded)" ON)
OPTION(GATE_USE_OWN_CLHEP "If 'ON', Gate does not use the standard CLHEP of GEANT4" ON)
OPTION(GATE_USE_CUDA "Gate use CUDA for primaries generation" OFF)

IF(GATE_USE_ROOT)
   FIND_PACKAGE(ROOT REQUIRED)
   INCLUDE_DIRECTORIES(${ROOT_INCLUDE_DIR})
   SET(LINK_DIRECTORIES ${ROOT_LIBRARY_DIR})
   #MESSAGE("Found Root in ${ROOT_INCLUDE_DIR} ${ROOT_LIBRARY_DIR}")
   IF(ROOT_INCLUDE_DIR)
   ELSE() 
      MESSAGE("Root Not found ! Please install Root and make it available in PATH")   
   ENDIF()
ENDIF()

#=========================================================
# Option for Optical Photon use

IF(GATE_USE_OPTICAL)
	FIND_PACKAGE(XML2 REQUIRED)
	STRING(REGEX REPLACE "-I" "" XML2_INCLUDE_DIR ${XML2_INCLUDE_DIR})
	INCLUDE_DIRECTORIES(${XML2_INCLUDE_DIR})
	SET(LINK_DIRECTORIES ${XML2_LIBRARIES})
	#MESSAGE("Found Xml2 in ${XML2_INCLUDE_DIR} ${XML2_LIBRARY_DIR} ${XML2_LIBRARIES}")
ENDIF(GATE_USE_OPTICAL)

#=========================================================
# Option for Clhep use

IF(GATE_USE_OWN_CLHEP)
   FIND_PACKAGE(CLHEP REQUIRED)
   INCLUDE_DIRECTORIES(${CLHEP_INCLUDE_DIR})
   SET(LINK_DIRECTORIES ${CLHEP_LIBRARIES})
   #MESSAGE("Found Clhep in ${CLHEP_INCLUDE_DIR} ${CLHEP_LIBRARY}")
ENDIF(GATE_USE_OWN_CLHEP)

#=========================================================
# Option for Cuda use

IF(GATE_USE_CUDA)
    FIND_PACKAGE(CUDA)
ENDIF(GATE_USE_CUDA)

# GATE_USE_LMF
# GATE_USE_ECAT7
# G4ANALYSIS_USE_ROOT_PLOTTER

#=========================================================
# Create configuration file
CONFIGURE_FILE(GateConfiguration.h.in ${PROJECT_BINARY_DIR}/GateConfiguration.h)
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})

#=========================================================
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
INCLUDE(${Geant4_USE_FILE})
INCLUDE_DIRECTORIES(
	${PROJECT_SOURCE_DIR}/source/arf/include
	${PROJECT_SOURCE_DIR}/source/geometry/include
	${PROJECT_SOURCE_DIR}/source/physics/include
	${PROJECT_SOURCE_DIR}/source/digits_hits/include
	${PROJECT_SOURCE_DIR}/source/general/include)

#=========================================================
# Locate sources and headers for this project
FILE(GLOB sources
	${PROJECT_SOURCE_DIR}/source/arf/src/*.cc
	${PROJECT_SOURCE_DIR}/source/geometry/src/*.cc
	${PROJECT_SOURCE_DIR}/source/digits_hits/src/*.cc
	${PROJECT_SOURCE_DIR}/source/physics/src/*.cc
	${PROJECT_SOURCE_DIR}/source/general/src/*.cc)
FILE(GLOB sourcesGPU
	${PROJECT_SOURCE_DIR}/source/gpu/src/track_gpu.cu)
FILE(GLOB headers
	${PROJECT_SOURCE_DIR}/source/arf/include/*.hh
	${PROJECT_SOURCE_DIR}/source/geometry/include/*.hh
	${PROJECT_SOURCE_DIR}/source/physics/include/*.hh
	${PROJECT_SOURCE_DIR}/source/digits_hits/include/*.hh
	${PROJECT_SOURCE_DIR}/source/general/include/*.hh
	${PROJECT_SOURCE_DIR}/source/gpu/include/*.hh)

#=========================================================
# Add the executable, and link it to the Geant4 libraries
IF(GATE_USE_CUDA AND CUDA_FOUND)
    SET(CUDA_NVCC_FLAGS "-arch;compute_20;-use_fast_math;-w")
    CUDA_ADD_EXECUTABLE(Gate Gate.cc ${sources} ${sourcesGPU} ${headers})
ELSE(GATE_USE_CUDA AND CUDA_FOUND)
    ADD_EXECUTABLE(Gate Gate.cc ${sources} ${headers})
ENDIF(GATE_USE_CUDA AND CUDA_FOUND)

TARGET_LINK_LIBRARIES(Gate ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${CLHEP_LIBRARIES} ${XML2_LIBRARIES})


#=========================================================
INSTALL(TARGETS Gate DESTINATION bin)
