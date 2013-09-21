#ifndef FMINCL_FREUDENSTEIN_ROTH_HPP_
#define FMINCL_FREUDENSTEIN_ROTH_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class freudenstein_roth : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    freudenstein_roth() : sum_square<BackendType>("Freudenstein",2,2,0){
        sum_square<BackendType>::local_minima_.push_back(48.98425);
    }
    void init(VectorType & X) const
    {
        X[0] = 0.5;
        X[1] = -2;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const{
        get(res,0,0) = 1;
        get(res,0,1) = - 3*V[1]*V[1] + 10*V[1] - 2;
        get(res,1,0) = 1;
        get(res,1,1) = 3*V[1]*V[1] + 2*V[1] - 14;
    }
    void fill_ym(const VectorType &V, ScalarType *res) const{
        res[0] = -13 + V[0] + (- V[1]*V[1] + 5*V[1] - 2)*V[1];
        res[1] = -29 + V[0] + (V[1]*V[1] + V[1] - 14)*V[1];
    }
};
#endif
