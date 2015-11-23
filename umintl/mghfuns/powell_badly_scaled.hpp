/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_MGHFUNS_POWELL_BADLY_SCALED_HPP_
#define UMINTL_MGHFUNS_POWELL_BADLY_SCALED_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class powell_badly_scaled : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::get;
public:
    powell_badly_scaled() : sum_square<BackendType>("Powell badly Scaled",2,2,0){ }
    void init(VectorType & X) const
    {
        X[0] = 0;
        X[1] = 1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        get(res,0,0) = std::pow(10,4)*V[1];
        get(res,0,1) = std::pow(10,4)*V[0];
        get(res,1,0) = -std::exp(-V[0]);
        get(res,1,1) = -std::exp(-V[1]);
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        res[0] = std::pow(10,4)*V[0]*V[1]-1;
        res[1] = std::exp(-V[0]) + std::exp(-V[1]) - 1.0001;
    }
};

#endif
