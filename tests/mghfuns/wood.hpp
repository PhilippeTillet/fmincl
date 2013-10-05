#ifndef FMINCL_WOOD_HPP_
#define FMINCL_WOOD_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;

template<class BackendType>
class wood : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::N_;
    using base_type::M_;
    using base_type::global_minimum_;
    using base_type::get;
public:
    wood() : base_type("Wood",6,4,0){ }
    void init(VectorType & X) const
    {
        X[0] = -3;
        X[1] = 1;
        X[2] = -3;
        X[3] = 1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        get(res,0,0) = -1*2*10*V[0];
        get(res,0,1) = 10;
        get(res,1,0) = -1;
        get(res,2,2) = -1*2*sqrt(90)*V[2];
        get(res,2,3) = sqrt(90);
        get(res,3,2) = -1;
        get(res,4,1) = sqrt(10);
        get(res,4,3) = sqrt(10);
        get(res,5,1) = (ScalarType)1/sqrt(10);
        get(res,5,2) = -1*(ScalarType)1/sqrt(10);
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        res[0] = 10*(V[1] - V[0]*V[0]);
        res[1] = 1 - V[0];
        res[2] = sqrt(90)*(V[3] - V[2]*V[2]);
        res[3] = 1 - V[2];
        res[4] = sqrt(10)*(V[1] + V[3] - 2);
        res[5] = (ScalarType)1/sqrt(10)*(V[1] - V[3]);
    }
};

#endif
