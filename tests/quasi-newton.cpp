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

    typedef typename get_backend<float>::type backend_float;
    typedef typename get_backend<double>::type backend_double;

    int result = EXIT_SUCCESS;

    std::cout << "Testing Float:" << std::endl;
    result |= test_option<backend_float>("BFGS", optimization_options(new quasi_newton(new bfgs())));

    std::cout << "Testing Double:" << std::endl;
    result |= test_option<backend_double>("BFGS", optimization_options(new quasi_newton(new bfgs())));

    return result;

}
