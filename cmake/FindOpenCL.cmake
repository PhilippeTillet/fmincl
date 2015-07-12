#Ideas for finding libOpenCL
set(ANDROID_CL_GLOB_HINTS /opt/adreno-driver*/lib)
set(X86_CL_GLOB_HINTS /opt/AMDAPPSDK*/lib/x86_64)

#OpenCL Hints
set(L_HINTS)
if(ANDROID)
    foreach(PATH ${ANDROID_GLOB_HINTS})
        file(GLOB _TMP ${PATH})
        set(L_HINTS ${L_HINTS} ${_TMP})
    endforeach()
else()
    foreach(PATH ${X86_GLOB_HINTS})
        file(GLOB _TMP ${PATH})
        set(L_HINTS ${L_HINTS} ${_TMP})
    endforeach()
    set(L_HINTS ${L_HINTS} ${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib/)
endif()

find_library(OPENCL_LIBRARIES NAMES OpenCL NO_CMAKE_FIND_ROOT_PATH HINTS ${L_HINTS} )
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenCL  DEFAULT_MSG OPENCL_LIBRARIES)
mark_as_advanced(OpenCL)
