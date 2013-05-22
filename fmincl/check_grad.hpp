/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CHECK_GRAD_HPP_
#define FMINCL_CHECK_GRAD_HPP_

#include <viennacl/vector.hpp>

namespace fmincl{

    template<class FUN>
    void check_grad(FUN const & fun, viennacl::vector<double> const & x0){
        unsigned int dim = x0.size();
        viennacl::vector<double> x(x0);
        viennacl::vector<double> fgrad(dim);
        viennacl::vector<double> numgrad(dim);
        double eps = 1e-8;
        fun(x,&fgrad);
        for(unsigned int i=0 ; i < dim ; ++i){
            double old = x(i);
            x(i) = old-eps; double vleft = fun(x,NULL);
            x(i) = old+eps; double vright = fun(x,NULL);
            numgrad(i) = (vright-vleft)/(2*eps);
        }
        std::cout << numgrad - fgrad << std::endl;

    }
}


#endif
