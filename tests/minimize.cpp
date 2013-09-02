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

typedef float ScalarType;
typedef Eigen::Matrix<ScalarType,Eigen::Dynamic,1> VectorType;
static const int dim = 2;

int main(){
    rosenbrock<ScalarType> fun;
    srand(time(NULL));
    VectorType X0(dim);
    for(unsigned int i = 0 ; i < dim ; ++i) X0(i) = 0.01*(double)rand()/RAND_MAX;
    //fmincl::check_grad(fun,X0);

    fmincl::optimization_options options;
    options.direction = new fmincl::quasi_newton_tag(new fmincl::lbfgs_tag());
    options.max_iter = 1e4;
    options.verbosity_level = 2;
    VectorType X =  fmincl::minimize<fmincl::backend::EigenTypes<ScalarType> >(fun,X0, options);

    //std::cout << "Minimum : " << X << std::endl;
}
