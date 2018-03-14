# - Try to find libxrl
# Once done this will define
#  LIBXRL_FOUND - System has libxrl
#  LIBXRL_INCLUDE_DIRS - The libxrl include directories
#  LIBXRL_LIBRARIES - The libraries needed to use libxrl
#  LIBXRL_DEFINITIONS - Compiler switches required for using libxrl

find_package(PkgConfig)
pkg_check_modules(PC_LIBXRL QUIET libxrl)
set(LIBXRL_DEFINITIONS ${PC_LIBXRL_CFLAGS_OTHER})

find_path(LIBXRL_INCLUDE_DIR xraylib.h
    HINTS ${PC_LIBXRL_INCLUDEDIR} ${PC_LIBXRL_INCLUDE_DIRS}
    PATH_SUFFIXES xraylib )

find_library(LIBXRL_LIBRARY NAMES xrl
    HINTS ${PC_LIBXRL_LIBDIR} ${PC_LIBXRL_LIBRARY_DIRS} )

set(LIBXRL_LIBRARIES ${LIBXRL_LIBRARY} )
set(LIBXRL_INCLUDE_DIRS ${LIBXRL_INCLUDE_DIR} )

include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set LIBXRL_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(libxrl  DEFAULT_MSG
    LIBXRL_LIBRARY LIBXRL_INCLUDE_DIR)

mark_as_advanced(LIBXRL_INCLUDE_DIR LIBXRL_LIBRARY )
