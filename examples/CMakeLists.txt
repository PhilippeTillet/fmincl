foreach(PROG rosenbrock mnist)
    add_executable(${PROG} ${PROG}.cpp)
    target_link_libraries(${PROG} openblas)
endforeach()

#add_executable(neural-network-mnist neural-network-mnist.cpp)

