/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include "fmincl/backends/eigen.hpp"
#include "directions_impl.hpp"

int main(){
    srand(0);
    float epsilon_float = 1e-3;
    double epsilon_double = 1e-5;

    std::cout << "====================" << std::endl;
    std::cout << "      Testing       "  << std::endl;
    std::cout << "     Directions     " << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Float:" << std::endl;
    std::cout << "-------------------" << std::endl;
    if(run_test<fmincl::backend::eigen_types<float> >(epsilon_float)==EXIT_FAILURE)
        return EXIT_FAILURE;

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Double" << std::endl;
    std::cout << "-------------------" << std::endl;
    if(run_test<fmincl::backend::eigen_types<double> >(epsilon_double)==EXIT_FAILURE)
        return EXIT_FAILURE;

    std::cout << std::endl;
    std::cout << "====================" << std::endl;
    std::cout << "Test passed!" << std::endl;
    return EXIT_SUCCESS;
}
