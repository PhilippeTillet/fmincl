/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_PENALTY1_HPP_
#define UMINTL_PENALTY1_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"


template<class BackendType>
class penalty1 : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    static const double a;
    using base_type::N_;
    using base_type::M_;
    using base_type::global_minimum_;
    using base_type::get;
public:
    penalty1(std::size_t n) : base_type("Penalty 1",n+1,n,0){
        if(n==4)
            global_minimum_ = 2.24997e-5;
        else if(n==10)
            global_minimum_ = 7.08765e-5;
        else
            throw "Not allowed dimensionality";
    }
    void init(VectorType & X) const
    {
        for(std::size_t n = 0 ; n < N_ ; ++n)
            X[n] = n;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < N_ ; ++m)
            RES(m,m) = sqrt(a);
        for(std::size_t n = 0 ; n < N_ ; ++n)
            RES(N_,n) = 2*V[n];


    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < N_ ; ++m)
            res[m] = sqrt(a)*(V[m]-1);
        ScalarType sum = 0;
        for(std::size_t n = 0 ; n < N_ ; ++n)
            sum+=V[n]*V[n];
        res[N_] = sum - 0.25;
    }
};

template<class BackendType>
const double penalty1<BackendType>::a = 1e-5;
#endif
