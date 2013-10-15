/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef FMINCL_BROWN_BADLY_SCALED_HPP_
#define FMINCL_BROWN_BADLY_SCALED_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class brown_badly_scaled : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    brown_badly_scaled() : base_type("Brown Badly Scaled",3,2,0){ }
    void init(VectorType & X) const
    {
        X[0] = 1;
        X[1] = 1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        get(res,0,0) = 1; get(res,0,1) = 0;
        get(res,1,0) = 0; get(res,1,1) = 1;
        get(res,2,0) = V[1]; get(res,2,1) = V[0];

    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        res[0] = V[0] - 1e6;
        res[1] = V[1] - 2e-6;
        res[2] = V[0]*V[1]-2;
    }
};

#endif
