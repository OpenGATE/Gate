#=========================================================
CMAKE_MINIMUM_REQUIRED(VERSION 3.12.0 FATAL_ERROR)
CMAKE_POLICY(SET CMP0012 NEW) # if() recognizes numbers and boolean constants.
CMAKE_POLICY(SET CMP0025 NEW) # Compiler id for Apple Clang is now AppleClang.
CMAKE_POLICY(SET CMP0057 NEW) # Support new if() IN_LIST operator.
CMAKE_POLICY(SET CMP0072 NEW) # FindOpenGL prefers GLVND by default when available.

#=========================================================
PROJECT(Gate)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules ${CMAKE_MODULE_PATH})

#=========================================================
# Configure Dart testing support.  This should be done before any
# MESSAGE(FATAL_ERROR ...) commands are invoked.
OPTION(BUILD_TESTING "Build the testing tree." OFF)
INCLUDE(CTest)

IF(BUILD_TESTING)
  ENABLE_TESTING()
  SET(BUILDNAME "${BUILDNAME}" CACHE STRING "Name of build on the dashboard")
  MARK_AS_ADVANCED(BUILDNAME)
ENDIF(BUILD_TESTING)
#=========================================================

#=========================================================
# Help function to debug CMake
MACRO (DD in)
  MESSAGE(${in}=[${${in}}])
ENDMACRO(DD)

#=========================================================
# Do not use RPATH for Mac, it fails to run Gate when install it
IF (APPLE)
  SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)
ENDIF()

#=========================================================
# Default build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

#=========================================================
# Native compilation
OPTION(GATE_COMPILE_WITH_NATIVE "Build with -march=native (works if build type is Release)" OFF)
IF(UNIX AND GATE_COMPILE_WITH_NATIVE)
  string(FIND ${CMAKE_CXX_FLAGS_RELEASE} -march=native pos_march_cxx)
  string(FIND ${CMAKE_C_FLAGS_RELEASE} -march=native pos_march_c)
  IF(${pos_march_cxx} EQUAL -1)
    SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
  ENDIF() 
  IF(${pos_march_c} EQUAL -1)
    SET(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -march=native")
  ENDIF()
ENDIF(UNIX AND GATE_COMPILE_WITH_NATIVE)

#=========================================================
# Option to use visualisation feature of G4
OPTION(GATE_USE_GEANT4_UIVIS "Build example with Geant4 UI and Vis drivers" ON)
IF(GATE_USE_GEANT4_UIVIS)
  FIND_PACKAGE(Geant4 REQUIRED ui_all vis_all)
ELSE()
  FIND_PACKAGE(Geant4 REQUIRED)
ENDIF()
set(CMAKE_CXX_STANDARD ${Geant4_CXX_STANDARD})

#=========================================================
# Refuse to build with MT geant4, allow dev override
IF(Geant4_multithreaded_FOUND)
    MESSAGE(WARNING "GATE should be compiled with a non-multithreaded installation of Geant4")
ENDIF()
#OPTION(GATE_ALLOW_MT_GEANT4 "Allow multithreaded geant4 (for MT Gate development)" OFF)
#IF(Geant4_multithreaded_FOUND)
    #IF(GATE_ALLOW_MT_GEANT4)
        #MESSAGE(WARNING "GATE does not yet work correctly with multithreaded Geant4, proceed with caution...")
    #ELSE()
        #MESSAGE(FATAL_ERROR "GATE should be compiled with a non-multithreaded installation of Geant4")
    #ENDIF()
#ENDIF()

# Check if OpenGL headers are still available
IF(Geant4_qt_FOUND OR Geant4_vis_opengl_x11_FOUND)
  FIND_PACKAGE(OpenGL QUIET)
  INCLUDE_DIRECTORIES(${OPENGL_INCLUDE_DIR})
  SET(GATE_USE_OPENGL ${OPENGL_FOUND})
ENDIF()

#=========================================================
# Build variables for G4 Version

STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\1" G4VERSION_MAJOR ${Geant4_VERSION})
STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\2" G4VERSION_MINOR ${Geant4_VERSION})
STRING(REGEX REPLACE "(.*)[.](.*)[.](.*)" "\\3" G4VERSION_PATCH ${Geant4_VERSION})
#MESSAGE(${Geant4_VERSION})
#MESSAGE(${G4VERSION_MAJOR})
#MESSAGE(${G4VERSION_MINOR})
#MESSAGE(${G4VERSION_PATCH})
IF(NOT ${G4VERSION_MAJOR}.${G4VERSION_MINOR} EQUAL 11.0)
  MESSAGE("Warning! This GATE version is not validated for Geant4 ${G4VERSION_MAJOR}.${G4VERSION_MINOR} distribution. Please use Geant4 11.0 distribution instead.")
ENDIF()

#=========================================================
# Option for Root use
SET(GATE_USE_ROOT ON)
SET(GATE_ANALYSIS_USE_GENERAL ON)
SET(GATE_ANALYSIS_USE_FILE ON)

OPTION(GATE_USE_OPTICAL "Gate use optical photon" ON)
OPTION(GATE_USE_XRAYLIB "Option for XRAYLIB use" OFF)
OPTION(GATE_USE_DAVIS "Gate use Davis LUT model" OFF)
OPTION(GATE_USE_LMF "Gate use LMF" OFF)
OPTION(GATE_USE_ECAT7 "Gate use ECAT7" OFF)
OPTION(GATE_USE_SYSTEM_CLHEP "If 'ON', Gate does not use the standard CLHEP of GEANT4. Use OFF if you compile G4 with embedded CLHEP" OFF)

#=========================================================
# Option for libTorch
OPTION(GATE_USE_TORCH "Gate use libTorch" OFF)

#=========================================================
# RTK / ITK
option(GATE_USE_RTK "Use the Reconstruction Toolkit (RTK, requires also ITK)" OFF)
option(GATE_USE_ITK "Use the Insight Toolkit (ITK, required by RTK, DICOM reader and thermal therapy capability)" OFF)

#=========================================================
# ROOT
IF(GATE_USE_ROOT)
  FIND_PACKAGE(ROOT REQUIRED)
  include(${ROOT_USE_FILE})
  get_target_property(ROOT_INTERFACE_COMPILE_FEATURES ROOT::Core INTERFACE_COMPILE_FEATURES)
  IF(NOT "${ROOT_INTERFACE_COMPILE_FEATURES}" STREQUAL cxx_std_${Geant4_CXX_STANDARD})
    MESSAGE(FATAL_ERROR "ROOT has been compiled with ${ROOT_INTERFACE_COMPILE_FEATURES}, please use cxx_std_${Geant4_CXX_STANDARD} as your Geant4.")
  ENDIF()
ENDIF()

#=========================================================
# Option for Optical Photon use
IF(GATE_USE_OPTICAL)
  FIND_PACKAGE(LibXml2 REQUIRED)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
  SET(LINK_DIRECTORIES ${LIBXML2_LIBRARIES})
ENDIF(GATE_USE_OPTICAL)

#=========================================================
# Option for XRAYLIB use
IF(GATE_USE_XRAYLIB)
  FIND_PACKAGE(LIBXRL REQUIRED)
  INCLUDE_DIRECTORIES(${LIBXRL_INCLUDE_DIR})
  SET(LINK_DIRECTORIES ${LIBXRL_LIBRARIES})
  IF(NOT LIBXRL_INCLUDE_DIR)
    MESSAGE("xraylib not found! Please install xraylib and make it available in PATH or manually set the LIBXRL_LIBRARY and LIBXRL_INCLUDE variables.")
  ENDIF()
ENDIF(GATE_USE_XRAYLIB)

#=========================================================
# Option for Clhep use
IF(GATE_USE_SYSTEM_CLHEP)
  FIND_PACKAGE(CLHEP REQUIRED)
  INCLUDE_DIRECTORIES(${CLHEP_INCLUDE_DIR})
  SET(LINK_DIRECTORIES ${CLHEP_LIBRARIES})
  MESSAGE("Your current CLHEP version is ${CLHEP_VERSION}")
  MESSAGE("Please refer to http://geant4.web.cern.ch/geant4/support/ReleaseNotes4.${G4VERSION_MAJOR}.${G4VERSION_MINOR}.html to verify that your CLHEP version is supported by Geant4.")

  #MESSAGE("Found Clhep in ${CLHEP_INCLUDE_DIR} ${CLHEP_LIBRARY}")
ENDIF()

#=========================================================
# Option for LMF use
IF(GATE_USE_LMF)
  ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/source/externals/lmf ${PROJECT_BINARY_DIR}/lmf)
  SET(LMF_LIBRARY "-L${PROJECT_BINARY_DIR}/lmf -lLMF")
  INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/source/externals/lmf ${PROJECT_SOURCE_DIR}/source/externals/lmf/lmf_v3.0/includes )
ENDIF(GATE_USE_LMF)
#IF(GATE_USE_LMF)
#  FIND_PATH(LMF_HOME lmf /usr /usr/local)
#  SET(LMF_INCLUDE_DIR "${LMF_HOME}/includes")
#  SET(LMF_LIBRARIES "${LMF_HOME}/lib")
#  SET(LMF_LIBRARY "-L${LMF_LIBRARIES} -lLMF")
#  INCLUDE_DIRECTORIES(${LMF_INCLUDE_DIR})
#  SET(LINK_DIRECTORIES ${LMF_LIBRARIES})
#  #MESSAGE("Found Lmf in ${LMF_INCLUDE_DIR} ${LMF_LIBRARIES} ${LMF_LIBRARY}")
#ENDIF(GATE_USE_LMF)

#=========================================================
# Option for ECAT7 use
IF(GATE_USE_ECAT7)
  ADD_SUBDIRECTORY(${PROJECT_SOURCE_DIR}/source/externals/ecat ${PROJECT_BINARY_DIR}/ecat)
  SET(ECAT7_LIBRARY "-L${PROJECT_BINARY_DIR}/ecat -lecat")
  INCLUDE_DIRECTORIES( ${PROJECT_SOURCE_DIR}/source/externals/ecat/src )
ENDIF(GATE_USE_ECAT7)
#IF(GATE_USE_ECAT7)
#  FIND_PATH(ECAT7_HOME ecat /usr /usr/local)
#  SET(ECAT7_INCLUDE_DIR "${ECAT7_HOME}/include")
#  SET(ECAT7_LIBRARIES "${ECAT7_HOME}/lib")
#  SET(ECAT7_LIBRARY "-L${ECAT7_LIBRARIES} -lecat")
#  INCLUDE_DIRECTORIES(${ECAT7_INCLUDE_DIR})
#  SET(LINK_DIRECTORIES ${ECAT7_LIBRARIES})
#  #MESSAGE("Found Ecat in ${ECAT7_INCLUDE_DIR} ${ECAT7_LIBRARIES} ${ECAT7_LIBRARY}")
#ENDIF(GATE_USE_ECAT7)

#=========================================================
# Create configuration file
CONFIGURE_FILE(GateConfiguration.h.in ${PROJECT_BINARY_DIR}/GateConfiguration.h)
INCLUDE_DIRECTORIES(${PROJECT_BINARY_DIR})

#=========================================================
# Setup Geant4 include directories and compile definitions
# Setup include directory for this project
INCLUDE(${Geant4_USE_FILE})
INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/source/io/include
  ${PROJECT_SOURCE_DIR}/source/arf/include
  ${PROJECT_SOURCE_DIR}/source/geometry/include
  ${PROJECT_SOURCE_DIR}/source/physics/include
  ${PROJECT_SOURCE_DIR}/source/digits_hits/include
  ${PROJECT_SOURCE_DIR}/source/general/include
  ${PROJECT_SOURCE_DIR}/source/externals/clhep/include)

#=========================================================
# Locate sources and headers for this project
FILE(GLOB sources
  ${PROJECT_SOURCE_DIR}/source/io/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/arf/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/geometry/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/digits_hits/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/physics/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/general/src/*.cc
  ${PROJECT_SOURCE_DIR}/source/externals/clhep/src/CLHEP/Matrix/*.cc
  ${PROJECT_SOURCE_DIR}/source/externals/clhep/src/CLHEP/RandomObjects/*.cc)

FILE(GLOB headers
  ${PROJECT_SOURCE_DIR}/source/io/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/arf/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/geometry/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/physics/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/digits_hits/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/general/include/*.hh
  ${PROJECT_SOURCE_DIR}/source/externals/clhep/include/*.hh)

#=========================================================
IF(GATE_USE_TORCH)
  FIND_PACKAGE(Torch REQUIRED)
  INCLUDE_DIRECTORIES(${TORCH_INCLUDE_DIRS})
ENDIF()

#=========================================================
# RTK
IF(GATE_USE_RTK OR GATE_USE_ITK)
  SET(ITK_MINIMAL_VERSION 5.1)
  SET(ITK_COMPONENTS ITKIOImageBase;ITKGDCM;ITKIOGDCM;ITKThresholding;ITKFFT;ITKImageCompose;ITKSmoothing)
  IF(GATE_USE_RTK)
    LIST(APPEND ITK_COMPONENTS RTK)
    SET(ITK_MINIMAL_VERSION 4.12)
  ENDIF()
  FIND_PACKAGE(ITK ${ITK_MINIMAL_VERSION} REQUIRED COMPONENTS ${ITK_COMPONENTS})
  IF(GATE_USE_RTK AND "${RTK_VERSION_MAJOR}.${RTK_VERSION_MINOR}" VERSION_LESS 2.0)
    MESSAGE(FATAL_ERROR "RTK version 2.0 is required")
  ENDIF()
  INCLUDE(${ITK_USE_FILE})
  STRING(REPLACE "-pedantic" "" CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
ELSE()
  ADD_SUBDIRECTORY(source/externals/itk-mhd ${PROJECT_BINARY_DIR}/itk-mhd)
  SET(ITK_LIBRARIES MetaIO itkzlib)
  INCLUDE_DIRECTORIES(
    ${PROJECT_SOURCE_DIR}/source/externals/itk-mhd/
    ${PROJECT_SOURCE_DIR}/source/externals/itk-mhd/itkzlib
    ${PROJECT_SOURCE_DIR}/source/externals/itk-mhd/src
    ${PROJECT_BINARY_DIR}/itk-mhd/)
  SET(ITKMHD_LIBS MetaIO itkzlib)
ENDIF()

#=========================================================
# Add the executable, and link it to the Geant4/ROOT/CLHEP/ITK libraries
ADD_LIBRARY(GateLib OBJECT ${sources} ${headers})
TARGET_LINK_LIBRARIES(GateLib ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${CLHEP_LIBRARIES} ${LIBXML2_LIBRARIES} ${LIBXRL_LIBRARIES} ${LMF_LIBRARY} ${ECAT7_LIBRARY} ${TORCH_LIBRARIES} ${ITK_LIBRARIES} pthread)
ADD_EXECUTABLE(Gate Gate.cc $<TARGET_OBJECTS:GateLib> )
TARGET_LINK_LIBRARIES(Gate GateLib)

#=========================================================
INSTALL(TARGETS Gate DESTINATION bin)

OPTION(GATE_COMPILE_GATEDIGIT "Build GateDigit tools" OFF)
IF(GATE_COMPILE_GATEDIGIT)
    ADD_EXECUTABLE(GateDigit_singles_sorter ${PROJECT_SOURCE_DIR}/source/bin/GateDigit_singles_sorter.cc $<TARGET_OBJECTS:GateLib> )
    ADD_EXECUTABLE(GateDigit_hits_digitizer ${PROJECT_SOURCE_DIR}/source/bin/GateDigit_hits_digitizer.cc  $<TARGET_OBJECTS:GateLib>)
    ADD_EXECUTABLE(GateDigit_coincidence_processor ${PROJECT_SOURCE_DIR}/source/bin/GateDigit_coincidence_processor.cc $<TARGET_OBJECTS:GateLib>)
    ADD_EXECUTABLE(GateDigit_seqCoinc2Cones ${PROJECT_SOURCE_DIR}/source/bin/GateDigit_seqCoinc2Cones.cc $<TARGET_OBJECTS:GateLib>)
    ADD_EXECUTABLE(Convert_CCMod2PETCoinc ${PROJECT_SOURCE_DIR}/source/bin/Convert_CCMod2PETCoinc.cc $<TARGET_OBJECTS:GateLib>)
    TARGET_LINK_LIBRARIES(GateDigit_singles_sorter GateLib)
    TARGET_LINK_LIBRARIES(GateDigit_hits_digitizer GateLib)
    TARGET_LINK_LIBRARIES(GateDigit_coincidence_processor GateLib)
    TARGET_LINK_LIBRARIES(GateDigit_seqCoinc2Cones GateLib)
    TARGET_LINK_LIBRARIES(Convert_CCMod2PETCoinc GateLib)
    INSTALL(TARGETS GateDigit_singles_sorter DESTINATION bin)
    INSTALL(TARGETS GateDigit_hits_digitizer DESTINATION bin)
    INSTALL(TARGETS GateDigit_coincidence_processor DESTINATION bin)
    INSTALL(TARGETS GateDigit_seqCoinc2Cones DESTINATION bin)
    INSTALL(TARGETS Convert_CCMod2PETCoinc DESTINATION bin)
ENDIF(GATE_COMPILE_GATEDIGIT)

#=========================================================
# We remove the warning option "shadow", because there are tons of
# such warning related to clhep/g4 system of units.
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")
#=========================================================

#=========================================================
# Display Geant4 version
MESSAGE("Your current Geant4 version is ${Geant4_VERSION}")

#=========================================================
# Add gprof to RELWITHDEBINFO target
IF(NOT APPLE)
  SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -pg")
  SET(CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_EXE_LINKER_FLAGS_RELWITHDEBINFO} -pg")
  SET(CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO "${CMAKE_SHARED_LINKER_FLAGS_RELWITHDEBINFO} -pg")
ENDIF()
#=========================================================

