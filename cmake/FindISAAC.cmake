
find_path(ISAAC_INCLUDE_DIR isaac/array.h
          HINTS ~/Development/isaac/include/ )

set(ISAAC_SEARCH_PATHS ~/Development/isaac/build/lib/ /lib/ /lib64/  /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/isaac/lib $ENV{ISAAC_HOME}/lib )
find_library(ISAAC_LIBRARY NAMES isaac PATHS ${ISAAC_SEARCH_PATHS})

set(ISAAC_LIBRARIES ${ISAAC_LIBRARY} )
set(ISAAC_INCLUDE_DIRS ${ISAAC_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(isaac  DEFAULT_MSG
                                  ISAAC_LIBRARY ISAAC_INCLUDE_DIR)


mark_as_advanced(isaac)
