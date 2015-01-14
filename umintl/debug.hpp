/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DEBUG_HPP
#define UMINTL_DEBUG_HPP


#include "tools/shared_ptr.hpp"
#include "atidlas/array.h"
#include "umintl/model_base.hpp"
#include <iostream>

#include <cmath>

namespace umintl{

template<class FUN>
double check_grad(FUN & fun, atidlas::array const & x0, double h)
{
    atidlas::array x(x0);
    atidlas::array fgrad(N, x.dtype());
    atidlas::array dummy(N, x.dtype());
    atidlas::array numgrad(N, x.dtype());
    double vl, vr;
    umintl::deterministic model;
    fun(x,vl,fgrad,model.get_value_gradient_tag());
    for(unsigned int i=0 ; i < N ; ++i)
    {
        double vx = x[i];
        x[i] = vx-h;
        fun(x,vl,dummy,model.get_value_gradient_tag());
        x[i] = vx+h;
        fun(x,vr,dummy,model.get_value_gradient_tag());
        numgrad[i] = (vr-vl)/(2*h);
        x[i]=vx;
    }
    return atidlas::max(max(abs(numgrad), abs(fgrad)) / abs(numgrad - fgrad));
}

//template<class BackendType, class FUN>
//double check_grad_variance(FUN & fun, atidlas::array const & x0, std::size_t N){
//
//

//    atidlas::array var(N);
//    atidlas::array tmp(N);
//    atidlas::array Hv(N);
//    BackendType::set_to_value(var,0,N);
//    c.fun().compute_hv_product(x0,c.g(),c.g(),Hv,hessian_vector_product(STOCHASTIC,S,offset_));

//    for(std::size_t i = 0 ; i < S ; ++i){
//        //tmp = (grad(xi) - grad(X)).^2
//        //var += tmp
//        c.fun().compute_hv_product(x0,c.g(),c.g(),tmp,hessian_vector_product(STOCHASTIC,1,offset_+i));
//        for(std::size_t i = 0 ; i < N ; ++i)
//            var[i]+=std::pow(tmp[i]-Hv[i],2);
//    }
//    BackendType::scale(N,(double)1/(S-1),var);
//    for(std::size_t i = 0 ; i < N ; ++i)
//        std::cout << var[i] << " ";
//    std::cout << std::endl;

//    c.fun().compute_hv_product_variance(x0,c.g(), var, hv_product_variance(STOCHASTIC,S,offset_));

//    for(std::size_t i = 0 ; i < N ; ++i)
//        std::cout << var[i] << " ";
//    std::cout << std::endl;

//    BackendType::delete_if_dynamically_allocated(var);
//    BackendType::delete_if_dynamically_allocated(tmp);
//    BackendType::delete_if_dynamically_allocated(Hv);
//}

}

#endif
