#ifndef FMINCL_BIGGS_EXP6_HPP_
#define FMINCL_BIGGS_EXP6_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class biggs_exp6 : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
    using base_type::M_;
    using base_type::N_;
    using base_type::get;
public:
    biggs_exp6() : base_type("Biggs EXP6",13,6,5.65565e-3){ }
    void init(VectorType & X) const
    {
        X[0] = 1;
        X[1] = 2;
        X[2] = 1;
        X[3] = 1;
        X[4] = 1;
        X[5] = 1;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            double t = 0.1*(m+1);
            RES(m,0) = -t*V[2]*exp(-t*V[0]);
            RES(m,1) = t*V[3]*exp(-t*V[1]);
            RES(m,2) = exp(-t*V[0]);
            RES(m,3) = -exp(-t*V[1]);
            RES(m,4) = -t*V[5]*exp(-t*V[4]);
            RES(m,5) =  exp(-t*V[4]);

        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < M_ ; ++m){
            double t = 0.1*(m+1);
            double y = exp(-t) - 5*exp(-10*t) + 3*exp(-4*t);
            res[m] = V[2]*exp(-t*V[0]) - V[3]*exp(-t*V[1]) + V[5]*exp(-t*V[4]) - y;
        }
    }
};

#endif
