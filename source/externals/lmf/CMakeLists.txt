project(LMF)
include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/lmf_v3.0/includes ${ROOT_INCLUDE_DIR})
file(GLOB csources
  ${PROJECT_SOURCE_DIR}/lmf_v3.0/src/*.c)
file(GLOB ccsources
  ${PROJECT_SOURCE_DIR}/lmf_v3.0/src/*.cc)
add_library(LMF STATIC ${csources} ${ccsources})
set_property(TARGET LMF PROPERTY C_STANDARD 90)
#set_property(TARGET LMF PROPERTY CXX_STANDARD 98)
target_compile_features(LMF PRIVATE cxx_constexpr cxx_generalized_initializers)
target_compile_definitions(LMF PRIVATE _FILE_OFFSET_BITS=64 PRIVATE _64 PRIVATE PIC)
target_link_libraries(LMF ${ROOT_LIBRARIES} )
