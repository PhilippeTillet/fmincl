/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_MGHFUNS_BROWN_DENNIS_HPP_
#define UMINTL_MGHFUNS_BROWN_DENNIS_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;

template<class BackendType>
class brown_dennis : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    static const double a;
    using base_type::N_;
    using base_type::M_;
    using base_type::global_minimum_;
    using base_type::get;
public:
    brown_dennis() : base_type("Brown-Dennis",20,4,85822.2){ }
    void init(VectorType & X) const
    {
        X[0] = 25;
        X[1] = 5;
        X[2] = -5;
        X[3] = -1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            ScalarType t = (ScalarType)(m+1)/5;
            double a = V[0] + t*V[1] - exp(t);
            double b = V[2] + V[3]*sin(t) - cos(t);
            get(res,m,0) = 2*a;
            get(res,m,1) = 2*t*a;
            get(res,m,2) = 2*b;
            get(res,m,3) = 2*sin(t)*b;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            ScalarType t = (ScalarType)(m+1)/5;
            double a = V[0] + t*V[1] - exp(t);
            double b = V[2] + V[3]*sin(t) - cos(t);
            res[m] = pow(a,2) + pow(b,2);
        }
    }
};

template<class BackendType>
const double powell_singular<BackendType>::a = 1e-5;
#endif
