add_executable(rosenbrock-openblas rosenbrock-openblas.cpp)
target_link_libraries(rosenbrock-openblas openblas)

add_executable(neural-network-mnist neural-network-mnist.cpp)

