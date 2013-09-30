/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include <iostream>

#include "test-common.hpp"

using namespace fmincl;

int main(){
    srand(0);
    int result = EXIT_SUCCESS;
    typedef typename get_backend<double>::type BackendType;

    result |= test_option("lbfgs [Double, M=2]", new quasi_newton<BackendType>(new lbfgs<BackendType>(2)));
    result |= test_option("lbfgs [Double, M=4]", new quasi_newton<BackendType>(new lbfgs<BackendType>(4)));
    result |= test_option("lbfgs [Double, M=8]", new quasi_newton<BackendType>(new lbfgs<BackendType>(8)));
    result |= test_option("lbfgs [Double, M=32]", new quasi_newton<BackendType>(new lbfgs<BackendType>(32)));

    return result;

}
