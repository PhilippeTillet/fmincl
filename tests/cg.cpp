/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


//#define VIENNACL_WITH_OPENCL

#include <cstdlib>
#include "viennacl/vector.hpp"
#include "fmincl/minimize.hpp"
#include "fmincl/utils.hpp"
#include "obj_fun.hpp"

typedef double NumericT;
static const int dim = 100;

int main(){
    rosenbrock<NumericT> fun;
    srand(time(NULL));
    viennacl::vector<NumericT> X0(dim); for(unsigned int i = 0 ; i < dim ; ++i) X0(i) = 0*(double)rand()/RAND_MAX;
    //fmincl::check_grad(fun,X0);

    fmincl::optimization_options options;

    options.direction = fmincl::direction::cg<fmincl::direction::polak_ribiere, fmincl::direction::no_restart>();
    options.line_search = fmincl::line_search::strong_wolfe_powell(1e-4,0.1,1.4);
    options.max_iter = 2000;

    viennacl::vector<NumericT> X =  fmincl::minimize(fun,X0, options);

    std::cout << "Minimum : " << X << std::endl;
    std::cout << fun.f_eval << " " << fun.df_eval << std::endl;
}
