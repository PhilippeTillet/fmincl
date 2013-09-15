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

//    std::cout << "[Polak-Ribière]:" << std::endl;
//    std::cout << " No Restart / Float:" << std::endl;
//    result |= test_option<backend_float>("Conjugate Gradient [Polak-Ribière - No Restart]", optimization_options(new conjugate_gradient(new polak_ribiere(), new no_restart())));
//    std::cout << "No Restart / Double:" << std::endl;
//    result |= test_option<backend_double>("Conjugate Gradient [Polak-Ribière - No Restart]", optimization_options(new conjugate_gradient(new polak_ribiere(), new no_restart())));
//    std::cout << " Restart K=D / Float:" << std::endl;
//    result |= test_option<backend_float>("Conjugate Gradient [Polak-Ribière - Restart K=D]", optimization_options(new conjugate_gradient(new polak_ribiere(), new restart_on_dim())));
//    std::cout << " Restart K=D / Double:" << std::endl;
//    result |= test_option<backend_double>("Conjugate Gradient [Polak-Ribière - Restart K=D]", optimization_options(new conjugate_gradient(new polak_ribiere(), new restart_on_dim())));

    std::cout << "---------------" << std::endl;

    std::cout << "[Fletcher-Reeves]:" << std::endl;
    std::cout << " No Restart / Float:" << std::endl;
    result |= test_option<backend_float>("Conjugate Gradient [Fletcher-Reeves - No Restart]", optimization_options(new conjugate_gradient(new fletcher_reeves(), new no_restart())));
//    std::cout << "No Restart / Double:" << std::endl;
//    result |= test_option<backend_double>("Conjugate Gradient [Fletcher-Reeves - No Restart]", optimization_options(new conjugate_gradient(new fletcher_reeves(), new no_restart())));
//    std::cout << " Restart K=D / Float:" << std::endl;
//    result |= test_option<backend_float>("Conjugate Gradient [Fletcher-Reeves - Restart K=D]", optimization_options(new conjugate_gradient(new fletcher_reeves(), new restart_on_dim())));
//    std::cout << " Restart K=D / Double:" << std::endl;
//    result |= test_option<backend_double>("Conjugate Gradient [Fletcher-Reeves - Restart K=D]", optimization_options(new conjugate_gradient(new fletcher_reeves(), new restart_on_dim())));

    return result;

}
