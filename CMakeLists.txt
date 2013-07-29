#=========================================================
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
#=========================================================


#=========================================================
PROJECT(DosiGateExamples)
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
    FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake-modules ${CMAKE_MODULE_PATH})
#=========================================================


#=========================================================
# Help function to debug CMake
MACRO (DD in)
  MESSAGE(${in}=[${${in}}])
ENDMACRO(DD)
#=========================================================


#=========================================================
INCLUDE(cmake-modules/GateExternalData.cmake)
INCLUDE(ExternalData)
#=========================================================


#=========================================================
# List of data to download
GateAddExampleData("DATA{brachytherapy/data/ct-2mm.raw}")
GateAddExampleData("DATA{external-beam-therapy-photon/data/patient-2mm.raw}")
#=========================================================


#=========================================================
# Target to download data (must be after GateAddExampleData)
ExternalData_add_target(GateExampleData)
#=========================================================
