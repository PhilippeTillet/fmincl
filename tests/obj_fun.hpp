/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_TESTS_OBJ_FUN
#define FMINCL_TESTS_OBJ_FUN

#include "fmincl/backend.hpp"

template<class ScalarType>
struct quad_fun{
    ScalarType operator()(fmincl::backend::VECTOR_TYPE const & x, fmincl::backend::VECTOR_TYPE * grad) const {
        ScalarType res = fmincl::backend::inner_prod(x,x);
        if(grad) *grad = static_cast<ScalarType>(2)*x;
        return res;
    }
};

template<class ScalarType>
struct rosenbrock{
    ScalarType operator()(fmincl::backend::VECTOR_TYPE const & x, fmincl::backend::VECTOR_TYPE * grad) const {
        ScalarType res=0;
        unsigned int dim = x.size();
#ifdef VIENNACL_WITH_OPENCL
        std::vector<ScalarType> x_cpu(dim);
        viennacl::copy(x, x_cpu);
        viennacl::backend::finish();
#else
        fmincl::backend::VECTOR_TYPE const & x_cpu = x;
#endif
        for(unsigned int i=0 ; i<dim-1;++i){
            res = res + 100*(pow(x_cpu[i+1] - x_cpu[i]*x_cpu[i],2)) + pow(1 - x_cpu[i],2);
        }
        if(grad){
#ifdef VIENNACL_WITH_OPENCL
            std::vector<ScalarType> grad_cpu(dim);
#else
            fmincl::backend::VECTOR_TYPE & grad_cpu = *grad;
#endif
            grad_cpu[0] = -400*x_cpu[0]*(x_cpu[1] - pow(x_cpu[0],2)) - 2*(1 - x_cpu[0]);
            for(unsigned int i=1 ; i<dim-1 ; ++i){
                double xi = x_cpu[i];
                double xim1 = x_cpu[i-1];
                double xip1 = x_cpu[i+1];
                grad_cpu[i] = 200*(xi - xim1*xim1) - 400*xi*(xip1-xi*xi) - 2*(1 - xi);
            }
            grad_cpu[dim-1] = 200*(x_cpu[dim-1]-x_cpu[dim-2]*x_cpu[dim-2]);
#ifdef VIENNACL_WITH_OPENCL
            viennacl::copy(grad_cpu, *grad);
            viennacl::backend::finish();
#endif
        }
        return res;
    }
};

#endif
