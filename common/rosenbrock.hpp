#ifndef FMINCL_ROSENBROCK_HPP_
#define FMINCL_ROSENBROCK_HPP_

#include <cmath>

template<class BackendType>
class rosenbrock{
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
public:
    static const std::size_t N = 10;

    static ScalarType true_minimum_value() { return 0; }

    ScalarType operator()(VectorType const & x, VectorType * grad) const {
        ScalarType res=0;
        VectorType const & x_cpu = x;
        for(unsigned int i=0 ; i<N-1;++i){
            res = res + 100*(pow(x_cpu[i+1] - x_cpu[i]*x_cpu[i],2)) + pow(1 - x_cpu[i],2);
        }
        if(grad){
            VectorType & grad_cpu = *grad;
            grad_cpu[0] = -400*x_cpu[0]*(x_cpu[1] - pow(x_cpu[0],2)) - 2*(1 - x_cpu[0]);
            for(unsigned int i=1 ; i<N-1 ; ++i){
                double xi = x_cpu[i];
                double xim1 = x_cpu[i-1];
                double xip1 = x_cpu[i+1];
                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
            }
            grad_cpu[N-1] = 200*(x_cpu[N-1]-x_cpu[N-2]*x_cpu[N-2]);
        }
        return res;
    }

};

#endif
