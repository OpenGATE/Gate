get_filename_component(_GateExternalData_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
include(${_GateExternalData_DIR}/ExternalData.cmake)

set(ExternalData_SOURCE_ROOT ${CMAKE_SOURCE_DIR})
set(ExternalData_BINARY_ROOT ${CMAKE_SOURCE_DIR})

set(ExternalData_TIMEOUT_INACTIVITY 0)
set(ExternalData_TIMEOUT_ABSOLUTE 0)

set(ExternalData_URL_TEMPLATES "" CACHE STRING
  "Additional URL templates for the ExternalData CMake script to look for testing data. E.g.
file:///var/bigharddrive/%(algo)/%(hash)")
mark_as_advanced(ExternalData_URL_TEMPLATES)
list(APPEND ExternalData_URL_TEMPLATES
  # Data used to be published by MIDAS
  # "https://midas3.kitware.com/midas/api/rest?method=midas.bitstream.download&checksum=%(hash)&algorithm=%(algo)"
  # New data server at KitWare, with new RESTful API, which only supports sha512 hashes (no md5)
  # To look at the collection (not super informative), go to:
  # https://data.kitware.com/#collection/5be2bffb8d777f21798e28bb
  # If you want to add data, ask David S. or David B.
  "https://data.kitware.com/api/v1/file/hashsum/%(algo)/%(hash)/download"
  )

function(GateAddBenchmarkData)
    ExternalData_expand_arguments(GateBenchmarkData dummy ${ARGN})
endfunction()
function(GateAddExampleData)
    ExternalData_expand_arguments(GateExampleData dummy ${ARGN})
endfunction()
