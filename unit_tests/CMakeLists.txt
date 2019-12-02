cmake_minimum_required(VERSION 3.0)

project(cmake_catch_test)

FIND_PACKAGE(Geant4 REQUIRED)
INCLUDE(${Geant4_USE_FILE})

# temporarily copied from main CMakeLists.txt file
# we should get a better CLHEP install so that this is no longer necessary
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-shadow")

# Prepare "Catch" library for other executables
set(CATCH_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../external/catch)
add_library(Catch INTERFACE)
target_include_directories(Catch INTERFACE ${CATCH_INCLUDE_DIR})

# test labels
# fast: less than 10 seconds per test case (on a fast computer)
# slow: more than 10 seconds per test case

# define which test modules to include
set(TEST_MODULES_LIST physics digits_hits general)
# todo: create map of tests properties to access later in the loop?

foreach(NAME IN LISTS TEST_MODULES_LIST)
    set(TARGET ${NAME}_tests)
    file(GLOB TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}/*.cc) # todo: replace with better method
    add_executable(${TARGET} ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp ${CMAKE_CURRENT_SOURCE_DIR}/utils/src/Stat.cc ${TEST_SRCS})
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/utils/include) # todo: move to sources directory? (out of unit_tests)
    target_link_libraries(${TARGET} Catch GateImpl ${Geant4_LIBRARIES} ${ROOT_LIBRARIES} ${EXTRA_GATE_LINK_LIBRARIES} )
    add_test(NAME ${TARGET}
             COMMAND ${TARGET} -d yes)
    set_tests_properties(${TARGET} PROPERTIES LABELS "fast;${NAME}" )
endforeach()
