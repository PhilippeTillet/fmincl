#ifndef FMINCL_VARIABLY_DIMENSIONED_HPP_
#define FMINCL_VARIABLY_DIMENSIONED_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class variably_dimensioned : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
public:
    variably_dimensioned(std::size_t n) : base_type("Variably Dimensioned",n+2,n,0){ }
    void init(VectorType & X) const
    {
        for(std::size_t n = 0 ; n < base_type::N_ ; ++n)
            X[n] = 1 - n/(ScalarType)base_type::N_;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        std::size_t N = base_type::N_;
        for(std::size_t m = 0 ; m < N ; ++m)
            for(std::size_t n = 0 ; n < N ; ++n)
                RES(m,n) = n;

        ScalarType sum = 0;
        for(std::size_t n = 0 ; n < N ; ++n)
            sum+=n*(V[n]-1);

        for(std::size_t n = 0 ; n < N ; ++n){
            RES(N,n) = n;
            RES(N+1,n) = 2*n*sum;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        std::size_t N = base_type::N_;
        for(std::size_t n = 0 ; n < base_type::N_ ; ++n){
            res[n] = V[n] - 1;
        }
        ScalarType sum = 0;
        for(std::size_t n = 0 ; n < N ; ++n)
            sum+=n*(V[n]-1);
        res[N] = sum;
        res[N+1] = sum*sum;
    }
};

#endif
