add_executable(directions-eigen-test directions-eigen.cpp)
add_test(directions-eigen directions-eigen-test)

add_executable(directions-cblas-test directions-cblas.cpp)
add_test(directions-cblas directions-cblas-test)
target_link_libraries(directions-cblas-test openblas)
