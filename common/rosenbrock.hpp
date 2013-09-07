#ifndef FMINCL_ROSENBROCK_HPP_
#define FMINCL_ROSENBROCK_HPP_

#include "Eigen/Dense"

template<class BackendType>
class rosenbrock{
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
public:
    rosenbrock(std::size_t N) : N_(N){ }

    ScalarType operator()(VectorType const & x, VectorType * grad) const {
        ScalarType res=0;
        VectorType const & x_cpu = x;
        for(unsigned int i=0 ; i<N_-1;++i){
            res = res + 100*(pow(x_cpu[i+1] - x_cpu[i]*x_cpu[i],2)) + pow(1 - x_cpu[i],2);
        }
        if(grad){
            VectorType & grad_cpu = *grad;
            grad_cpu[0] = -400*x_cpu[0]*(x_cpu[1] - pow(x_cpu[0],2)) - 2*(1 - x_cpu[0]);
            for(unsigned int i=1 ; i<N_-1 ; ++i){
                double xi = x_cpu[i];
                double xim1 = x_cpu[i-1];
                double xip1 = x_cpu[i+1];
                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
            }
            grad_cpu[N_-1] = 200*(x_cpu[N_-1]-x_cpu[N_-2]*x_cpu[N_-2]);
        }
        return res;
    }
private:
    std::size_t N_;

};

#endif
