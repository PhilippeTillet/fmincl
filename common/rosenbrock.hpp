#ifndef FMINCL_ROSENBROCK_HPP_
#define FMINCL_ROSENBROCK_HPP_

#include <cmath>
#include <vector>

template<std::size_t _N, class BackendType>
class rosenbrock{
    typedef typename BackendType::VectorType VectorType;
public:
    static const std::size_t N = _N;

    static double true_minimum_value() { return 0; }

    static void local_minima_value(std::vector<double> &) {  }

    static void init(VectorType & X){
        for(unsigned int i = 0 ; i < N ; i+=2){
            X[i] = -1.2;
            X[i+1] = 1;
        }
    }

    double operator()(VectorType const & x, VectorType * grad) const {
        double res=0;
        for(unsigned int i=0 ; i<N-1;++i){
            res = res + 100*(pow(x[i+1] - x[i]*x[i],2)) + pow(1 - x[i],2);
        }
        if(grad){
            VectorType & grad_cpu = *grad;
            grad_cpu[0] = -400*x[0]*(x[1] - pow(x[0],2)) - 2*(1 - x[0]);
            for(unsigned int i=1 ; i<N-1 ; ++i){
                double xi = x[i];
                double xim1 = x[i-1];
                double xip1 = x[i+1];
                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
            }
            grad_cpu[N-1] = 200*(x[N-1]-x[N-2]*x[N-2]);
        }
        return res;
    }

};

#endif
