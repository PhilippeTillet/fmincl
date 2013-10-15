/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_GAUSSIAN_HPP_
#define UMINTL_GAUSSIAN_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class gaussian : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    gaussian() : base_type("Gaussian",15,3,1.12793e-8){ }
    void init(VectorType & X) const
    {
        X[0] = 0.4;
        X[1] = 1;
        X[2] = 0;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            ScalarType t = (ScalarType)(8-(int)(m+1))/2;
            ScalarType e = exp(-0.5*V[1]*pow(t-V[2],2));
            get(res,m,0) = e;
            get(res,m,1) = -0.5*V[0]*pow(t-V[2],2)*e;
            get(res,m,2) = V[0]*V[1]*(t-V[2])*e;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        ScalarType y[15] = {0.0009, 0.0044, 0.0175, 0.0540, 0.1295, 0.2420, 0.3521, 0.3989,
                            0.3521, 0.2420, 0.1295, 0.0540, 0.0175, 0.0044, 0.0009};
        for(std::size_t m = 0 ; m < M_ ; ++m){
            ScalarType t = (ScalarType)(8-(int)(m+1))/2;
            ScalarType e = exp(-0.5*V[1]*pow(t-V[2],2));
            res[m] = V[0]*e - y[m];
        }
    }
};

#endif
