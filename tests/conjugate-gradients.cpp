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

    std::cout << "[Polak-Ribière]:" << std::endl;
    result |= test_option<float>("Conjugate Gradient [Float - Polak-Ribière - No Restart]", new conjugate_gradient(new polak_ribiere(), new no_restart()));
    result |= test_option<double>("Conjugate Gradient [Double - Polak-Ribière - No Restart]", new conjugate_gradient(new polak_ribiere(), new no_restart()));
    result |= test_option<float>("Conjugate Gradient [Float - Polak-Ribière - Restart K=D]", new conjugate_gradient(new polak_ribiere(), new restart_on_dim()));
    result |= test_option<double>("Conjugate Gradient [Double - Polak-Ribière - Restart K=D]", new conjugate_gradient(new polak_ribiere(), new restart_on_dim()));

//    std::cout << "---------------"<< std::endl;

//    std::cout << "[Fletcher-Reeves]:"<< std::endl;
//    result |= test_option<float>("Conjugate Gradient [Float - Fletcher-Reeves - No Restart]", new conjugate_gradient(new fletcher_reeves(), new no_restart()));
//    result |= test_option<double>("Conjugate Gradient [Double - Fletcher-Reeves - No Restart]", new conjugate_gradient(new fletcher_reeves(), new no_restart()));
//    result |= test_option<float>("Conjugate Gradient [Float - Fletcher-Reeves - Restart K=D]", new conjugate_gradient(new fletcher_reeves(), new restart_on_dim()));
//    result |= test_option<double>("Conjugate Gradient [Double - Fletcher-Reeves - Restart K=D]", new conjugate_gradient(new fletcher_reeves(), new restart_on_dim()));

    return result;

}
