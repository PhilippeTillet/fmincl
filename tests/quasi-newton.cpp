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

    std::cout << "Testing Float:" << std::endl;
    result |= test_option<float>("BFGS", new quasi_newton(new bfgs()));

    std::cout << std::endl;

    std::cout << "Testing Double:" << std::endl;
    result |= test_option<double>("BFGS", new quasi_newton(new bfgs()));

    return result;

}
