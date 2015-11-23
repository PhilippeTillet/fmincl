/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_GULF_HPP_
#define UMINTL_GULF_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class gulf : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    gulf(std::size_t M) : base_type("Gulf",M,3,0){ }
    void init(VectorType & X) const
    {
        X[0] = 5;
        X[1] = 2.5;
        X[2] = 0.15;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_; ++m){
            ScalarType t = (ScalarType)(m+1)/100;
            ScalarType y = 25+pow(-50*log(t),(double)2/3);
            ScalarType x1 = V[0], x2 = V[1], x3 = V[2];
            double p = pow(abs(y-x2),x3);
            ScalarType e = exp(-pow(abs(y-V[1]),V[2])/V[0]) ;
            RES(m,0) = p/(x1*x1)*e;
            RES(m,1) = x3*p/(x1*(y-x2))*e;
            RES(m,2) = -p*log(pow(x2-y,2))/(2*x1)*e;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_; ++m){
            ScalarType t = (ScalarType)(m+1)/100;
            ScalarType y = 25+pow(-50*log(t),(double)2/3);
            ScalarType e = exp(-pow(abs(y-V[1]),V[2])/V[0]) ;
            res[m] = e - t;
        }
    }
};

#endif
