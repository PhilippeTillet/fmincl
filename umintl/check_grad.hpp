/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_CHECK_GRAD_HPP
#define UMINTL_CHECK_GRAD_HPP

#include "tools/shared_ptr.hpp"
#include "tags.hpp"
#include <iostream>

#include <cmath>

namespace umintl{

    template<class BackendType, class FUN>
    typename BackendType::ScalarType check_grad(FUN & fun, typename BackendType::VectorType const & x0, std::size_t N, typename BackendType::ScalarType h){
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        VectorType x = BackendType::create_vector(N);
        BackendType::copy(N,x0,x);
        VectorType fgrad = BackendType::create_vector(N);
        VectorType dummy = BackendType::create_vector(N);
        VectorType numgrad = BackendType::create_vector(N);
        ScalarType res = 0;
        ScalarType vl, vr;
        fun(x,vl,fgrad,umintl::value_gradient_tag());
        for(unsigned int i=0 ; i < N ; ++i){
            ScalarType vx = x[i];
            x[i] = vx-h; fun(x,vl,dummy,umintl::value_gradient_tag());
            x[i] = vx+h; fun(x,vr,dummy,umintl::value_gradient_tag());
            numgrad[i] = (vr-vl)/(2*h);
            x[i]=vx;
        }
        for(unsigned int i=0 ; i < N ; ++i){
            ScalarType denom = std::max(std::fabs((double)numgrad[i]),std::abs((double)fgrad[i]));
            ScalarType diff = std::fabs(numgrad[i]-fgrad[i]);
            //std::cout << numgrad[i] << " " << fgrad[i] << std::endl;
            if(denom>1)
                diff/=denom;
            res = std::max(res,diff);
        }
        return res;
    }


}
#endif
