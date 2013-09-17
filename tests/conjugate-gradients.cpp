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

    result |= test_option<float>("Conjugate Gradient [Float - Polak-Ribière]", new conjugate_gradient(new polak_ribiere()));
    result |= test_option<double>("Conjugate Gradient [Double - Polak-Ribière]", new conjugate_gradient(new polak_ribiere()));
    result |= test_option<float>("Conjugate Gradient [Float - Fletcher-Reeves]", new conjugate_gradient(new fletcher_reeves()));
    result |= test_option<double>("Conjugate Gradient [Double - Fletcher-Reeves]", new conjugate_gradient(new fletcher_reeves()));

    return result;

}
