add_executable(rosenbrock rosenbrock.cpp)
target_link_libraries(rosenbrock ${ATIDLAS_LIBRARIES} ${OPENCL_LIBRARIES})

#add_executable(neural-network-mnist neural-network-mnist.cpp)
