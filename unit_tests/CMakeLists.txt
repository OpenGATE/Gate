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

include_directories(utils/include)

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

# Make physics tests executable
add_executable(Physics_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/physics/GateSourcePencilBeamTest.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/physics/GateSPSPosDistributionTest.cc  ${CMAKE_CURRENT_SOURCE_DIR}/utils/src/Stat.cc)
target_link_libraries(Physics_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
add_test(NAME Physics_tests
         COMMAND Physics_tests -d yes)
set_tests_properties(Physics_tests PROPERTIES LABELS "fast;physics" )

# Make digits_hits tests executable
add_executable(Digits_hits_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/digits_hits/GatePulseTest.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/src/Stat.cc)
target_link_libraries(Digits_hits_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
add_test(NAME Digits_hits_tests
        COMMAND Digits_hits_tests -d yes)
set_tests_properties(Digits_hits_tests PROPERTIES LABELS "fast;digits_hits" )

# Make general tests executable
add_executable(General_tests ${CMAKE_CURRENT_SOURCE_DIR}/catch/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/general/GateHounsfieldDensityTableTest.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/src/Stat.cc)
target_link_libraries(General_tests Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
add_test(NAME General_tests
        COMMAND General_tests -d yes)
set_tests_properties(General_tests PROPERTIES LABELS "fast;general" )