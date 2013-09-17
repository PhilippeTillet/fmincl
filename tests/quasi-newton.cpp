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

    result |= test_option<float>("BFGS [Float]", new quasi_newton(new bfgs()));
    result |= test_option<double>("BFGS [Double]", new quasi_newton(new bfgs()));

    return result;

}
