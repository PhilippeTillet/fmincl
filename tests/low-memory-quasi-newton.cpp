/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include <iostream>

#include "test-common.hpp"

using namespace umintl;

int main(){
    srand(0);
    int result = EXIT_SUCCESS;
    typedef typename get_backend<double>::type BackendType;

    result |= test_option("lbfgs [Double, M=2]", new low_memory_quasi_newton<BackendType>(2));
    result |= test_option("lbfgs [Double, M=4]", new low_memory_quasi_newton<BackendType>(4));
    result |= test_option("lbfgs [Double, M=8]", new low_memory_quasi_newton<BackendType>(8));
    result |= test_option("lbfgs [Double, M=32]", new low_memory_quasi_newton<BackendType>(32));

    return result;

}
