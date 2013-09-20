#ifndef FMINCL_ROSENBROCK_HPP_
#define FMINCL_ROSENBROCK_HPP_

#include <cmath>
#include <vector>
#include "sum_square.hpp"

template<class BackendType>
class rosenbrock : public sum_square<BackendType>{
    typedef sum_square<BackendType> base_type;
    typedef typename BackendType::VectorType VectorType;
    typedef double ScalarType;
public:
    rosenbrock(std::size_t n) : base_type("Rosenbrock",n,n,0){
        if(n%2>0)
            throw "Provide an even size for the rosenbrock function!";
    }
    void init(VectorType & X) const
    {
        for(unsigned int i = 0 ; i < base_type::N_ ; i+=2){
            X[i] = -1.2;
            X[i+1] = 1;
        }
    }
    void fill_dym_dxn(VectorType const & V, ScalarType * res) const
    {
        for(std::size_t m = 0 ; m < base_type::M_ ; m++)
            for(std::size_t n = 0 ; n < base_type::N_ ; n++)
                base_type::get(res,m,n) = 0;

        for(std::size_t n = 0 ; n < base_type::N_ ; n+=2){
            base_type::get(res,n,n) = -20*V[n];
            base_type::get(res,n,n+1) = 10;
            base_type::get(res,n+1,n) = -1;
        }
    }
    void fill_ym(VectorType const & V, ScalarType * res) const
    {
        for(unsigned int i = 0 ; i < base_type::N_ ; i+=2){
            res[i] = 10*(V[i+1] - V[i]*V[i]);
            res[i+1] = 1 - V[i];
        }
    }
};

//template<std::size_t _N, class BackendType>
//class rosenbrock{
//    typedef typename BackendType::VectorType VectorType;
//public:
//    static const std::size_t N = _N;

//    static double true_minimum_value() { return 0; }

//    static void local_minima_value(std::vector<double> &) {  }

//    static void init(VectorType & X){
//        for(unsigned int i = 0 ; i < N ; i+=2){
//            X[i] = -1.2;
//            X[i+1] = 1;
//        }
//    }

//    double operator()(VectorType const & x, VectorType * grad) const {
//        double res=0;
//        for(unsigned int i=0 ; i<N-1;++i){
//            res = res + 100*(pow(x[i+1] - x[i]*x[i],2)) + pow(1 - x[i],2);
//        }
//        if(grad){
//            VectorType & grad_cpu = *grad;
//            grad_cpu[0] = -400*x[0]*(x[1] - pow(x[0],2)) - 2*(1 - x[0]);
//            for(unsigned int i=1 ; i<N-1 ; ++i){
//                double xi = x[i];
//                double xim1 = x[i-1];
//                double xip1 = x[i+1];
//                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
//            }
//            grad_cpu[N-1] = 200*(x[N-1]-x[N-2]*x[N-2]);
//        }
//        return res;
//    }

//};

#endif
