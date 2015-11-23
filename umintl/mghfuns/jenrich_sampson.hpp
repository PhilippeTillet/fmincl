/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_JENRICH_SAMPSON_HPP_
#define UMINTL_JENRICH_SAMPSON_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class jenrich_sampson : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    jenrich_sampson() : base_type("Jenrich Sampson",10,2,124.36218){ }
    void init(VectorType & X) const
    {
        X[0] = 0;
        X[1] = 0;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 1 ; m < M_+1 ; ++m){
            get(res,m-1,0) = m*-std::exp(m*V[0]);
            get(res,m-1,1) = m*-std::exp(m*V[1]);
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 1 ; m < M_+1 ; ++m){
            res[m-1] = 2 + 2*m - (std::exp(m*V[0]) + std::exp(m*V[1]));
        }
    }
};

#endif
