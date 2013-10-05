/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>

#include "common/rosenbrock.hpp"

#include "umintl/backends/cblas.hpp"
#include "umintl/minimize.hpp"

#include "timer.hpp"

typedef float ScalarType;

void print_vector(ScalarType * x, std::size_t N){
    std::cout << "["; for(std::size_t i = 0 ; i < N ; ++i) std::cout << x[i] << ((i==N-1)?']':',') << std::flush;
}

int main(){
    srand(0);
    static const std::size_t N = rosenbrock<2,umintl::backend::cblas_types<ScalarType> >::N;

    std::cout << "====================" << std::endl;
    std::cout << "Minimization of the generalized Rosenbrock function" << std::endl;
    std::cout << "Dimension : " << N << std::endl;
    std::cout << "====================" << std::endl;

    ScalarType* X0 = new ScalarType[N];
    ScalarType* S = new ScalarType[N];

    for(std::size_t i = 0 ; i < N ; ++i) X0[i] = (ScalarType)rand()/RAND_MAX;
    std::cout << "Starting at : " << std::endl;
    print_vector(X0,N);
    std::cout << "--------------" << std::endl;

    umintl::optimization_options options;
    //options.direction = new umintl::quasi_newton_tag(new umintl::lbfgs_tag()); //You can select the number of storage pairs in the constructor of lbfgs_tag()
    options.direction = new umintl::quasi_newton(new umintl::bfgs()); //Uncomment for BFGS
    //options.direction = new umintl::cg_tag(); //Uncomment for ConjugateGradient
    options.max_iter = 1000;
    options.verbosity_level=2;
    options.stopping_criterion = new umintl::gradient_treshold(1e-4); //Stops when the gradient is below 1e-4
    umintl::minimize<umintl::backend::cblas_types<ScalarType> >(S,rosenbrock<2,umintl::backend::cblas_types<ScalarType> >(),X0,N,options);
    std::cout << "--------------" << std::endl;
    std::cout << "Solution : " << std::endl;
    print_vector(S,N);
    std::cout << std::endl;
    std::cout << "Test passed!" << std::endl;

    return EXIT_SUCCESS;
}
