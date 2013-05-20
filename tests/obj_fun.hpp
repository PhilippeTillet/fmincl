#ifndef FMINCL_TESTS_OBJ_FUN
#define FMINCL_TESTS_OBJ_FUN

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

template<class ScalarType>
struct quad_fun{
    ScalarType operator()(viennacl::vector<ScalarType> const & x, viennacl::vector<ScalarType> * grad) const {
        viennacl::scalar<ScalarType> res = viennacl::linalg::inner_prod(x,x);
        if(grad) *grad = static_cast<ScalarType>(2)*x;
        return res;
    }
};

template<class ScalarType>
struct rosenbrock{
    ScalarType operator()(viennacl::vector<ScalarType> const & x, viennacl::vector<ScalarType> * grad) const {
        ScalarType res=0;
        unsigned int dim = x.size();
        std::vector<ScalarType> x_cpu(dim);
        viennacl::copy(x, x_cpu);
        viennacl::backend::finish();
        for(unsigned int i=0 ; i<dim-1;++i){
            res = res + 100*(pow(x_cpu[i+1] - x_cpu[i]*x_cpu[i],2)) + pow(1 - x_cpu[i],2);
        }
        if(grad){
            std::vector<ScalarType> grad_cpu(dim);
            grad_cpu[0] = -400*x_cpu[0]*(x_cpu[1] - pow(x_cpu[0],2)) - 2*(1 - x_cpu[0]);
            for(unsigned int i=1 ; i<dim-1 ; ++i){
                double xi = x_cpu[i];
                double xim1 = x_cpu[i-1];
                double xip1 = x_cpu[i+1];
                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
            }
            grad_cpu[dim-1] = 200*(x_cpu[dim-1]-x_cpu[dim-2]*x_cpu[dim-2]);
            viennacl::copy(grad_cpu, *grad);
            viennacl::backend::finish();
        }
        return res;

    }
};

#endif
