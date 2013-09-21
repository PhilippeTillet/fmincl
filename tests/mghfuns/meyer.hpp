#ifndef FMINCL_MEYER_HPP_
#define FMINCL_MEYER_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class meyer : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
public:
    meyer() : base_type("Meyer",16,3,87.9458){ }
    void init(VectorType & X) const
    {
        X[0] = 0.02;
        X[1] = 4000;
        X[2] = 250;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t i = 1 ; i < base_type::M_+1 ; ++i){
            RES(i-1, 0) = exp(V[1]/(45+5*i+V[2]));
            RES(i-1, 1) = V[0]/(45+5*i+V[2])*exp(V[1]/(45+5*i+V[2]));
            RES(i-1, 2) = -V[0]*V[1]/pow(45+5*i+V[2],2)*exp(V[1]/(45+5*i+V[2]));
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        ScalarType y[16] = { 34780, 28610, 23650, 19630, 16370, 13720, 11540, 9744
                             , 8261, 7030, 6005, 5147, 4427, 3820, 3307, 2872 };
        for(std::size_t i = 1 ; i < base_type::M_+1 ; ++i){
            res[i-1] = V[0]*exp(V[1]/(45+5*i+V[2])) - y[i-1];
        }
    }
};

#endif
