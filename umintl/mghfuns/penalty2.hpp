/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_PENALTY2_HPP_
#define UMINTL_PENALTY2_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"


template<class BackendType>
class penalty2 : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    static const double a;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
    using base_type::global_minimum_;
public:
    penalty2(std::size_t n) : base_type("Penalty 2",2*n,n,0){
        if(n==4)
            global_minimum_ = 9.37629e-6;
        else if(n==10)
            global_minimum_ = 2.93660e-4;
        else
            throw "Not allowed dimensionality";
    }
    void init(VectorType & X) const
    {
        for(std::size_t n = 0 ; n < N_ ; ++n)
            X[n] = 0.5;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        RES(0,0) = 1;
        for(std::size_t i = 1 ; i < N_ ; ++i){
            RES(i,i) = sqrt(a)*0.1*exp(0.1*V[i]);
            RES(i,i-1) = sqrt(a)*0.1*exp(0.1*V[i-1]);
        }
        for(std::size_t i = N_ ; i < M_-1; ++i){
            RES(i,i-N_) = sqrt(a)*0.1*(exp(0.1*V[(i-N_)]));
        }
        for(std::size_t i = 0 ; i < N_ ; ++i)
            RES(M_-1,i)=2*(N_ - i)*V[i];

    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        res[0] = V[0] - 0.2;
        for(std::size_t i = 1 ; i < N_ ; ++i){
            ScalarType y = exp(0.1*(i+1)) + exp(0.1*i);
            res[i] = sqrt(a)*((exp(0.1*V[i]) + exp(0.1*V[i-1])) - y);
        }
        for(std::size_t i = N_ ; i < M_-1; ++i){
            res[i] = sqrt(a)*(exp(0.1*V[(i-N_)]) - exp(-0.1));
        }
        ScalarType sum = 0;
        for(std::size_t i = 0 ; i < N_ ; ++i)
            sum+=(N_ - i)*V[i]*V[i];
        res[M_-1] = sum - 1;

    }
};

template<class BackendType>
const double penalty2<BackendType>::a = 1e-5;
#endif
