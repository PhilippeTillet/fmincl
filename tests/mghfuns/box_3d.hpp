/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef FMINCL_BOX_3D_HPP_
#define FMINCL_BOX_3D_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class box_3d : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    box_3d() : base_type("Box 3D",10,3,0){ }
    void init(VectorType & X) const
    {
        X[0] = 0;
        X[1] = 10;
        X[2] = 20;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_; ++m){
            ScalarType t = 0.1*(m+1);
            RES(m,0) = -t*exp(-t*V[0]);
            RES(m,1) = t*exp(-t*V[1]);
            RES(m,2) = -(exp(-t) - exp(-10*t));
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_; ++m){
            ScalarType t = 0.1*(m+1);
            res[m] = exp(-t*V[0]) - exp(-t*V[1]) - V[2]*(exp(-t) - exp(-10*t));
        }
    }
};

#endif
