#ifndef FMINCL_WATSON_HPP_
#define FMINCL_WATSON_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

using namespace std;
#define RES(i,j) base_type::get(res,i,j)

template<class BackendType>
class watson : public sum_square<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
    typedef sum_square<BackendType> base_type;
public:
    watson(std::size_t n) : base_type("Watson",31,n,0){
        if(n==6)
            base_type::global_minimum_ = 2.28767e-3;
        else if(n==9)
            base_type::global_minimum_ = 1.39976e-3;
        else if(n==12)
            base_type::global_minimum_ = 4.72238e-3;
        else if(n==20)
            base_type::global_minimum_ = 2.48631e-3;
        else
            throw "Invalid dimensionality";

    }
    void init(VectorType & X) const
    {
        for(std::size_t n = 0 ; n < base_type::N_ ; ++n)
            X[n] = 0;
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < 29 ; ++m){
            ScalarType t = (m+1)/(ScalarType)29;

            ScalarType sum2 = 0;
            for(std::size_t j = 0 ; j < base_type::N_ ; ++j){
                sum2 += V[j]*pow(t,j);
            }

            RES(m,0) = -2*1*sum2;
            for(std::size_t n = 1 ; n < base_type::N_ ; ++n){
                RES(m,n) = n*pow(t,n-1) -  2*pow(t,n)*sum2;
            }
        }
        for(std::size_t n = 0 ; n < base_type::N_ ; ++n){
            RES(29,n) = 0;
            RES(30,n) = 0;
        }
        RES(29,0) = 1;
        RES(30,0) = -2*V[0];
        RES(30,1) = 1;

    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < 29 ; ++m){
            ScalarType t = (m+1)/(ScalarType)29;
            ScalarType sum1 = 0;
            for(std::size_t j = 1 ; j < base_type::N_ ; ++j){
                sum1 += j*V[j]*pow(t,j-1);
            }
            ScalarType sum2 = 0;
            for(std::size_t j = 0 ; j < base_type::N_ ; ++j){
                sum2 += V[j]*pow(t,j);
            }
            res[m] = sum1 - sum2*sum2 - 1;
        }
        res[29] = V[0];
        res[30] = V[1] - V[0]*V[0] - 1;
    }
};

#endif
