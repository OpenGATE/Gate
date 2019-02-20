cmake_minimum_required(VERSION 3.0)

project(cmake_catch_test)

FIND_PACKAGE(Geant4 REQUIRED)
INCLUDE(${Geant4_USE_FILE})

# temporarily copied from main CMakeLists.txt file
# we should get a better CLHEP install so that this is no longer necessary
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})


# test labels
# fast: less than 10 seconds per test case (on a fast computer)
# slow: more than 10 seconds per test case
# general: testing things defined in the source/general directory
# physics: testing things defined in the source/physics directory

# TODO: define a macro to condense the repetitive stuff

# Make HU builder test executable
# NOT YET
#add_executable(HU_builder_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/general/GateHounsfieldToMaterialBuilderTest.cc)
#add_definitions( -DGATE_SOURCE_DIR=${CMAKE_HOME_DIRECTORY} )
#target_link_libraries(HU_builder_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
#add_test(NAME HU_builder_tests
#COMMAND HU_builder_tests -d yes)
#set_tests_properties(HU_builder_tests PROPERTIES LABELS "slow;general" )

# Make HU density table test executable
# NOT YET
#add_executable(HU_density_table_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/general/GateHounsfieldDensityTableTest.cc)
#target_link_libraries(HU_density_table_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
#add_test(NAME HU_density_table_tests
#         COMMAND HU_density_table_tests -d yes)
#set_tests_properties(HU_density_table_tests PROPERTIES LABELS "fast;general" )

# Make pencil beam source test executable
add_executable(PB_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/physics/GateSourcePencilBeamTest.cc)
target_link_libraries(PB_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
add_test(NAME PB_tests
         COMMAND PB_tests -d yes)
set_tests_properties(PB_tests PROPERTIES LABELS "fast;physics" )
