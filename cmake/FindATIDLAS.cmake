
find_path(ATIDLAS_INCLUDE_DIR atidlas/array.h
          HINTS ~/Development/ATIDLAS/include/ )

set(ATIDLAS_SEARCH_PATHS ~/Development/ATIDLAS/build/lib/ /lib/ /lib64/  /usr/lib /usr/lib64 /usr/local/lib /usr/local/lib64 /opt/ATIDLAS/lib $ENV{ATIDLAS_HOME}/lib )
find_library(ATIDLAS_LIBRARY NAMES atidlas PATHS ${ATIDLAS_SEARCH_PATHS})

set(ATIDLAS_LIBRARIES ${ATIDLAS_LIBRARY} )
set(ATIDLAS_INCLUDE_DIRS ${ATIDLAS_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ATIDLAS  DEFAULT_MSG
                                  ATIDLAS_LIBRARY ATIDLAS_INCLUDE_DIR)


mark_as_advanced(ATIDLAS)
