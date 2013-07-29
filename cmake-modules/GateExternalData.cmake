get_filename_component(_GateExternalData_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${_GateExternalData_DIR}/ExternalData.cmake)

set(ExternalData_SOURCE_ROOT ${CMAKE_SOURCE_DIR})
set(ExternalData_BINARY_ROOT ${CMAKE_SOURCE_DIR})

set(ExternalData_URL_TEMPLATES "" CACHE STRING
  "Additional URL templates for the ExternalData CMake script to look for testing data. E.g.
file:///var/bigharddrive/%(algo)/%(hash)")
mark_as_advanced(ExternalData_URL_TEMPLATES)
list(APPEND ExternalData_URL_TEMPLATES
  # Data published by MIDAS
  "http://midas3.kitware.com/midas/api/rest?method=midas.bitstream.download&checksum=%(hash)&algorithm=%(algo)"
  )

function(GateAddExampleData)
  ExternalData_expand_arguments(GateExampleData dummy ${ARGN})
endfunction()
