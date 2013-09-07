add_executable(directions-eigen-test directions-eigen.cpp)
target_link_libraries(directions-eigen-test)
add_test(directions-eigen directions-eigen-test)

add_executable(directions-openblas-test directions-openblas.cpp)
target_link_libraries(directions-openblas-test)
add_test(directions-openblas directions-openblas-test)
target_link_libraries(directions-openblas-test openblas)
