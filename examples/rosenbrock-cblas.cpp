/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>

#include "common/rosenbrock.hpp"

#include "fmincl/backends/cblas.hpp"
#include "fmincl/minimize.hpp"

#include "timer.hpp"

typedef float ScalarType;

void print_vector(ScalarType * x, std::size_t N){
    std::cout << "["; for(std::size_t i = 0 ; i < N ; ++i) std::cout << x[i] << ((i==N-1)?']':',') << std::flush;
}

int main(){
    srand(0);
    unsigned int D = 10;

    std::cout << "====================" << std::endl;
    std::cout << "Minimization of the generalized Rosenbrock function" << std::endl;
    std::cout << "Dimension : " << D << std::endl;
    std::cout << "====================" << std::endl;

    ScalarType* X0 = new ScalarType[D];
    ScalarType* S = new ScalarType[D];

    for(std::size_t i = 0 ; i < D ; ++i) X0[i] = (ScalarType)rand()/RAND_MAX;
    std::cout << "Starting at : " << std::endl;
    print_vector(X0,D);
    std::cout << "--------------" << std::endl;

    fmincl::optimization_options options;
    //options.direction = new fmincl::quasi_newton_tag(new fmincl::lbfgs_tag()); //You can select the number of storage pairs in the constructor of lbfgs_tag()
    //options.direction = new fmincl::quasi_newton_tag(new fmincl::bfgs_tag()); //Uncomment for BFGS
    //options.direction = new fmincl::cg_tag(); //Uncomment for ConjugateGradient
    options.max_iter = 1000;
    options.verbosity_level=2;
    options.stopping_criterion = new fmincl::gradient_treshold(1e-4); //Stops when the gradient is below 1e-4
    fmincl::minimize<fmincl::backend::cblas_types<ScalarType> >(S,rosenbrock<fmincl::backend::cblas_types<ScalarType> >(D),X0,D,options);
    std::cout << "--------------" << std::endl;
    std::cout << "Solution : " << std::endl;
    print_vector(S,D);
    std::cout << std::endl;
    std::cout << "Test passed!" << std::endl;

    return EXIT_SUCCESS;
}
