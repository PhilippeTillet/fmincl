/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_ROSENBROCK_HPP_
#define UMINTL_ROSENBROCK_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class rosenbrock : public sum_square<BackendType>{
    typedef sum_square<BackendType> base_type;
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    rosenbrock(std::size_t n) : base_type("Rosenbrock",n,n,0){
        if(n%2>0)
            throw "Provide an even size for the rosenbrock function!";
    }
    void init(VectorType & X) const
    {
        for(unsigned int i = 0 ; i < N_ ; i+=2){
            X[i] = -1.2;
            X[i+1] = 1;
        }
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; m++)
            for(std::size_t n = 0 ; n < N_ ; n++)
                get(res,m,n) = 0;

        for(std::size_t n = 0 ; n < N_ ; n+=2){
            get(res,n,n) = -20*V[n];
            get(res,n,n+1) = 10;
            get(res,n+1,n) = -1;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(unsigned int i = 0 ; i < N_ ; i+=2){
            res[i] = 10*(V[i+1] - V[i]*V[i]);
            res[i+1] = 1 - V[i];
        }
    }
};

#endif
