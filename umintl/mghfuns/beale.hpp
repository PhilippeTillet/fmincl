/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_BEALE_HPP_
#define UMINTL_BEALE_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class beale : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    beale() : base_type("Beale",3,2,0){ }
    void init(VectorType & X) const
    {
        X[0] = 1;
        X[1] = 1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            get(res,m,0) = -1 + std::pow(V[1],m+1);
            get(res,m,1) = V[0]*(m+1)*std::pow(V[1],m);
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        ScalarType alpha[3] = {1.5, 2.25, 2.625};
        for(std::size_t m = 0 ; m < M_ ; ++m)
            res[m] = alpha[m]-V[0]+V[0]*std::pow(V[1],m+1);
    }
};

#endif
