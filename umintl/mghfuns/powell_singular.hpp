/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_POWELL_SINGULAR_HPP_
#define UMINTL_POWELL_SINGULAR_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;

template<class BackendType>
class powell_singular : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    static const double a;
    using base_type::N_;
    using base_type::M_;
    using base_type::global_minimum_;
    using base_type::get;
public:
    powell_singular(std::size_t n) : base_type("Powell Singular",n,n,0){
        if(n%4>0)
            throw "Invalid";
    }
    void init(VectorType & X) const
    {
        for(std::size_t i = 0 ; i < N_ ; i+=4){
            X[i+0] = 3;
            X[i+1] = -1;
            X[i+2] = 0;
            X[i+3] = 1;
        }
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m)
            for(std::size_t n = 0 ; n < N_ ; ++n)
                get(res,m,n) = 0;

        for(std::size_t i = 0 ; i < N_ ; i+=4){
            get(res,i+0,i+0) = 1;
            get(res,i+0,i+1) = 10;
            get(res,i+1,i+2) = sqrt(5);
            get(res,i+1,i+3) = -sqrt(5);
            get(res,i+2,i+1) = 2*(V[i+1] - 2*V[i+2]);
            get(res,i+2,i+2) = -4*(V[i+1] - 2*V[i+2]);
            get(res,i+3,i+0) = sqrt(10)*2*(V[i+0] - V[i+3]);
            get(res,i+3,i+3) = sqrt(10)*2*-1*(V[i+0] - V[i+3]);
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t i = 0 ; i < N_ ; i+=4){
            res[i+0] = V[i+0] + 10*V[i+1];
            res[i+1] = sqrt(5)*(V[i+2] - V[i+3]);
            res[i+2] = pow(V[i+1] - 2*V[i+2],2);
            res[i+3] = sqrt(10)*pow(V[i+0] - V[i+3],2);
        }
    }
};

#endif
