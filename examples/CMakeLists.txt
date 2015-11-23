foreach(PROG rosenbrock mnist data)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} openblas)
endforeach()

add_executable(data data.cpp)
target_link_libraries(data openblas)

#add_executable(neural-network-mnist neural-network-mnist.cpp)

