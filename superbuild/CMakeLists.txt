#=========================================================
cmake_minimum_required(VERSION 2.8)
cmake_policy(VERSION 2.8)
#=========================================================


#=========================================================
# set a default build type if it is undefined, then make sure it goes in the cache
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Build configuration type" FORCE)
set(build_type ${CMAKE_BUILD_TYPE})
#=========================================================


#=========================================================
MACRO (DD in)
  MESSAGE(${in}=${${in}})
ENDMACRO(DD)
#=========================================================


#=========================================================
project(SuperBuild_CREATIS)
include(ExternalProject)
#=========================================================


#=========================================================
SET(BUILD_GATE ON)
SET(GATE_BRANCH "master" CACHE STRING "Branch to build (master or develop)")
SET(BUILD_GATE_WITH_RTK OFF)
#OPTION(BUILD_GATE_WITH_RTK "Build Gate with RTK (branch hybrid)" OFF)
#=========================================================


#=========================================================
set(base "${CMAKE_BINARY_DIR}")
set_property(DIRECTORY PROPERTY EP_PREFIX ${base})
SET(INSTALL_PREFIX ${base}/install CACHE INTERNAL "Install directory" FORCE)
SET(SRC_PREFIX ${base}/src)
SET(CMAKE_INSTALL_PREFIX ${INSTALL_PREFIX} CACHE PATH "Install directory" FORCE)
#=========================================================


#=========================================================
find_package(Git REQUIRED)

#DD(BUILD_GATE)
#DD(BUILD_GATE_WITH_RTK)
#=========================================================


#=========================================================
# ITK
IF (BUILD_GATE_WITH_RTK)
  ExternalProject_Add(
    itk
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    GIT_REPOSITORY git://itk.org/ITK.git
    GIT_TAG v4.2.0
    CMAKE_ARGS
    -DCMAKE_C_FLAGS_DEBUG:STRING=${CMAKE_C_FLAGS_DEBUG}
    -DCMAKE_C_FLAGS_MINSIZEREL:STRING=${CMAKE_C_FLAGS_MINSIZEREL}
    -DCMAKE_C_FLAGS_RELEASE:STRING=${CMAKE_C_FLAGS_RELEASE}
    -DCMAKE_C_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_C_FLAGS_RELWITHDEBINFO}
    -DCMAKE_CXX_FLAGS_DEBUG:STRING=${CMAKE_CXX_FLAGS_DEBUG}
    -DCMAKE_CXX_FLAGS_MINSIZEREL:STRING=${CMAKE_CXX_FLAGS_MINSIZEREL}
    -DCMAKE_CXX_FLAGS_RELEASE:STRING=${CMAKE_CXX_FLAGS_RELEASE}
    -DCMAKE_CXX_FLAGS_RELWITHDEBINFO:STRING=${CMAKE_CXX_FLAGS_RELWITHDEBINFO}
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}
    -DCMAKE_BUILD_TYPE:STRING=${build_type}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DBUILD_EXAMPLES:BOOL=OFF
    -DBUILD_TESTING:BOOL=OFF
    )
ENDIF()
#=========================================================


#=========================================================
# CLHEP
# IF (BUILD_GATE)
#   ExternalProject_Add(
#     clhep
#     STAMP_DIR ${base}/stamp
#     DOWNLOAD_DIR ${base}/dl
#     INSTALL_DIR ${INSTALL_PREFIX}
#     URL http://proj-clhep.web.cern.ch/proj-clhep/DISTRIBUTION/tarFiles/clhep-2.1.3.1.tgz
#     CONFIGURE_COMMAND ${SRC_PREFIX}/clhep/CLHEP/configure --prefix=${INSTALL_PREFIX}
#     )
# ENDIF()
#=========================================================


#=========================================================
# ROOT
IF (BUILD_GATE)
  ExternalProject_Add(
    root
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    URL ftp://root.cern.ch/root/root_v5.34.05.source.tar.gz
    CONFIGURE_COMMAND ${SRC_PREFIX}/root/configure --prefix=${INSTALL_PREFIX}
    )
ENDIF()
#=========================================================


#=========================================================
# RTK
IF (BUILD_GATE_WITH_RTK)
  ExternalProject_Add(
    rtk
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    DEPENDS itk
    GIT_REPOSITORY https://github.com/SimonRit/RTK.git
    GIT_TAG origin/master
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${INSTALL_PREFIX}
    -DBUILD_APPLICATIONS:BOOL=ON
    -DBUILD_DOXYGEN:BOOL=OFF
    -DBUILD_SHARED_LIBS:BOOL=OFF
    -DBUILD_TESTING:BOOL=OFF
    -DITK_DIR:PATH=${INSTALL_PREFIX}
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}
    )
ENDIF()
#=========================================================


#=========================================================
# GEANT4
IF (BUILD_GATE)
  ExternalProject_Add(
    geant4
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    URL http://geant4.web.cern.ch/geant4/support/source/geant4.9.6.p03.tar.gz
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${INSTALL_PREFIX}
    -DGEANT4_INSTALL_DATA:BOOL=ON
    -DGEANT4_USE_OPENGL_X11:BOOL=ON
    -DGEANT4_USE_QT:BOOL=OFF
    -DGEANT4_USE_RAYTRACER_X11:BOOL=ON
    -DGEANT4_USE_SYSTEM_CLHEP:BOOL=OFF
    -DQT_QMAKE_EXECUTABLE:FILEPATH=${qmake_executable}
    -DCMAKE_BUILD_TYPE:STRING=${build_type}
    -DBUILD_SHARED_LIBS:BOOL=ON
    -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}
    )
ENDIF()
#=========================================================


#=========================================================
# GATE without RTK
IF (NOT BUILD_GATE_WITH_RTK)
  ExternalProject_Add(
    opengate
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    DEPENDS geant4 root
    GIT_REPOSITORY gitopengate:opengate.git
    GIT_TAG origin/${GATE_BRANCH}
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${INSTALL_PREFIX}
    -DGATE_ANALYSIS_USE_FILE:BOOL=ON
    -DGATE_ANALYSIS_USE_GENERAL:BOOL=ON
    -DGATE_USE_ECAT7:BOOL=OFF
    -DGATE_USE_GEANT4_UIVIS:BOOL=ON
    -DGATE_USE_GPU:BOOL=OFF
    -DGATE_USE_LMF:BOOL=OFF
    -DGATE_USE_OPTICAL:BOOL=ON
    -DGATE_USE_SYSTEM_CLHEP:BOOL=OFF
    -DGATE_USE_ROOT:BOOL=ON
    -DGATE_USE_RTK:BOOL=OFF
    -DGeant4_DIR:PATH=${INSTALL_PREFIX}
    -DROOTCINT_EXECUTABLE:FILEPATH=${INSTALL_PREFIX}/bin/rootcint
    -DGATE_USE_GEANT4_UIVIS:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}
    )
ENDIF()
#=========================================================


#=========================================================
# GATE
IF (BUILD_GATE_WITH_RTK)
  ExternalProject_Add(
    opengate
    STAMP_DIR ${base}/stamp
    DOWNLOAD_DIR ${base}/dl
    INSTALL_DIR ${INSTALL_PREFIX}
    DEPENDS geant4 clhep root itk rtk
    GIT_REPOSITORY gitopengate:opengate.git
    GIT_TAG origin/hybrid
    CMAKE_ARGS
    -DCMAKE_PREFIX_PATH:PATH=${INSTALL_PREFIX}
    -DGATE_ANALYSIS_USE_FILE:BOOL=ON
    -DGATE_ANALYSIS_USE_GENERAL:BOOL=ON
    -DGATE_USE_ECAT7:BOOL=OFF
    -DGATE_USE_GEANT4_UIVIS:BOOL=ON
    -DGATE_USE_GPU:BOOL=OFF
    -DGATE_USE_LMF:BOOL=OFF
    -DGATE_USE_OPTICAL:BOOL=OFF
    -DGATE_USE_SYSTEM_CLHEP:BOOL=ON
    -DCLHEP_CONFIG_EXECUTABLE:FILEPATH=${INSTALL_PREFIX}/bin/clhep-config
    -DCLHEP_INCLUDE_DIR:PATH=${INSTALL_PREFIX}/include
    -DGATE_USE_ROOT:BOOL=ON
    -DGATE_USE_RTK:BOOL=ON
    -DGeant4_DIR:PATH=${INSTALL_PREFIX}
    -DROOTCINT_EXECUTABLE:FILEPATH=${INSTALL_PREFIX}/bin/rootcint
    -DGATE_USE_GEANT4_UIVIS:BOOL=ON
    -DCMAKE_INSTALL_PREFIX:PATH=${INSTALL_PREFIX}
    -DRTK_DIR:PATH=${SRC_PREFIX}/rtk-build
    )
ENDIF()
#=========================================================
