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

    result |= test_option<float>("LBFGS [Float, M=1]", new quasi_newton(new lbfgs(1)));
    result |= test_option<float>("LBFGS [Float, M=2]", new quasi_newton(new lbfgs(2)));
    result |= test_option<float>("LBFGS [Float, M=8]", new quasi_newton(new lbfgs(8)));
    result |= test_option<float>("LBFGS [Float, M=32]", new quasi_newton(new lbfgs(32)));
    result |= test_option<double>("LBFGS [Double, M=1]", new quasi_newton(new lbfgs(1)));
    result |= test_option<double>("LBFGS [Double, M=2]", new quasi_newton(new lbfgs(2)));
    result |= test_option<double>("LBFGS [Double, M=8]", new quasi_newton(new lbfgs(8)));
    result |= test_option<double>("LBFGS [Double, M=32]", new quasi_newton(new lbfgs(32)));

    return result;

}
