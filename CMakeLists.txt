cmake_minimum_required(VERSION 2.8)

project(FMinCL)

set(VERSION_MAJOR 0)
set(VERSION_MINOR 0)

set(VERSION ${VERSION_MAJOR}.${VERSION_MINOR})

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(OpenCL QUIET REQUIRED)
include_directories(${OPENCL_INCLUDE_DIRS})

# User options
option(BUILD_TESTING "Build the tests " ON)
option(BUILD_EXAMPLES "Build the examples " ON)

# Includes
include_directories(BEFORE ${PROJECT_SOURCE_DIR})

# Add visibility of headers
# Necessary for Qt-Creator usage.
################
file( GLOB_RECURSE MAKE_HEADERS_VISIBLE_SRC
    *.hpp *.h)
add_custom_target( MAKE_HEADERS_VISIBLE SOURCES ${MAKE_HEADERS_VISIBLE_SRC} )


#Set Warning level
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic")
set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -msse4 -O3")

if(BUILD_EXAMPLES)
   add_subdirectory(examples)
endif()

if(BUILD_TESTING)
   INCLUDE(CTest)
   add_subdirectory(tests)
endif()

