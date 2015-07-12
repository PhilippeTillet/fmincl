cmake_minimum_required(VERSION 2.8)

project(UMinTL)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)

set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR})
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(ISAAC QUIET REQUIRED)
find_package(OpenCL QUIET REQUIRED)

# User options
option(BUILD_EXAMPLES "Build the examples" ON)
option(BUILD_TESTING "Build the tests " ON)

# Includes
include_directories(BEFORE ${PROJECT_SOURCE_DIR})
include_directories(${ISAAC_INCLUDE_DIRS})
# Add visibility of headers
# Necessary for Qt-Creator usage.
################
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC
    *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )


#Set Warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -msse4 -O3 -DNDEBUG")
#set(CMAKE_BUILD_TYPE Release)


if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(BUILD_TESTING)
   INCLUDE(CTest)
   add_subdirectory(tests)
endif()

