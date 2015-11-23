/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_MGHFUNS_TRIGONOMETRIC_HPP_
#define UMINTL_MGHFUNS_TRIGONOMETRIC_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"


template<class BackendType>
class trigonometric : public sum_square<BackendType>{
    typedef sum_square<BackendType> base_type;
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    trigonometric(std::size_t n) : base_type("Trigonometric",n,n,0){ }

    void init(VectorType & X) const
    {
        for(unsigned int i = 0 ; i < N_ ; i++)
            X[i] = 1/(ScalarType)N_;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; m++){
            for(std::size_t n = 0 ; n < N_ ; n++)
                base_type::get(res,m,n) = sin(V[n]);
            base_type::get(res,m,m) += (m+1)*sin(V[m]) - cos(V[m]);
        }

    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        ScalarType sum = 0;
        for(size_t i = 0 ; i < N_ ; ++i)
            sum+=cos(V[i]);
        for(size_t i = 0 ; i < N_ ; i++)
            res[i] = N_ - sum + (i+1)*(1 - cos(V[i])) - sin(V[i]);
    }
};

#endif
