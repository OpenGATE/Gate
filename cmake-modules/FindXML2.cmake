# - Finds XML2 instalation
# This module sets up XML2 information 
# It defines:
# XML2_FOUND          If the XML2 is found
# XML2_INCLUDE_DIR    PATH to the include directory
# XML2_LIBRARIES      Most common libraries
# XML2_LIBRARY_DIR    PATH to the library directory 


find_program(XML2_CONFIG_EXECUTABLE xml2-config
  PATHS $ENV{PATH}/bin)

if(NOT XML2_CONFIG_EXECUTABLE)
  set(XML2_FOUND FALSE)
else()    
  set(XML2_FOUND TRUE)

  execute_process(
    COMMAND ${XML2_CONFIG_EXECUTABLE} --prefix 
    OUTPUT_VARIABLE XML2SYS 
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND ${XML2_CONFIG_EXECUTABLE} --version 
    OUTPUT_VARIABLE XML2_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND ${XML2_CONFIG_EXECUTABLE} --cflags
    OUTPUT_VARIABLE XML2_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  execute_process(
    COMMAND ${XML2_CONFIG_EXECUTABLE} --libs
    OUTPUT_VARIABLE XML2_LIBRARIES
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  set(XML2_LIBRARY_DIR ${XML2SYS}/lib)

  if(NOT XML2_FIND_QUIETLY)
    message(STATUS "Found XML2 ${XML2_VERSION} in ${XML2SYS}")
  endif()
endif()
