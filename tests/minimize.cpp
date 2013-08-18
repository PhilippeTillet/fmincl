/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#include <cstdlib>
#include "fmincl/backend.hpp"
#include "fmincl/minimize.hpp"
#include "obj_fun.hpp"

typedef double NumericT;
static const int dim = 100;

int main(){
    rosenbrock<NumericT> fun;
    srand(time(NULL));
    fmincl::backend::VECTOR_TYPE X0(dim);
    for(unsigned int i = 0 ; i < dim ; ++i) X0(i) = 0.01*(double)rand()/RAND_MAX;
    //fmincl::check_grad(fun,X0);

    fmincl::wrapped_optimization_options options;
    options.direction = new fmincl::quasi_newton(new fmincl::bfgs());
    options.max_iter = 1e4;
    options.verbosity_level = 2;
    fmincl::backend::VECTOR_TYPE X =  fmincl::minimize(fun,X0, options);

    //std::cout << "Minimum : " << X << std::endl;
}
