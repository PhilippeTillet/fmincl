/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_UTILS_HPP
#define FMINCL_UTILS_HPP

#include <iostream>

#include "fmincl/backend.hpp"

namespace fmincl{

    namespace detail{

        struct function_wrapper{
            function_wrapper() : n_value_calc_(0), n_derivative_calc_(0){ }
            virtual double operator()(backend::VECTOR_TYPE const & x, backend::VECTOR_TYPE * grad) const = 0;
            unsigned int n_value_calc() const { return n_value_calc_; }
            unsigned int n_derivative_calc() const { return n_derivative_calc_; }
        protected:
            mutable unsigned int n_value_calc_;
            mutable unsigned int n_derivative_calc_;
        };

        template<class Fun>
        struct function_wrapper_impl : function_wrapper{
            function_wrapper_impl(Fun const & fun) : fun_(fun){ }
            double operator()(backend::VECTOR_TYPE const & x, backend::VECTOR_TYPE * grad) const {
                ++n_value_calc_;
                if(grad) ++n_derivative_calc_;
                return fun_(x, grad);
            }
        private:
            Fun const & fun_;
        };

        class state{
        public:
            state(backend::VECTOR_TYPE const & x0, detail::function_wrapper const & fun) : fun_(fun), iter_(0), dim_(x0.size()), x_(x0), g_(dim_), p_(dim_), xm1_(dim_), gm1_(dim_){

            }

            detail::function_wrapper const & fun() { return fun_; }
            unsigned int & iter() { return iter_; }
            unsigned int & dim() { return dim_; }
            backend::VECTOR_TYPE & x() { return x_; }
            backend::VECTOR_TYPE & g() { return g_; }
            backend::VECTOR_TYPE & xm1() { return xm1_; }
            backend::VECTOR_TYPE & gm1() { return gm1_; }
            backend::VECTOR_TYPE & p() { return p_; }
            double & val() { return valk_; }
            double & valm1() { return valkm1_; }
            double & diff() { return diff_; }
            double & dphi_0() { return dphi_0_; }

        private:
            detail::function_wrapper const & fun_;
            unsigned int iter_;
            unsigned int dim_;
            backend::VECTOR_TYPE x_;
            backend::VECTOR_TYPE g_;
            backend::VECTOR_TYPE p_;
            backend::VECTOR_TYPE xm1_;
            backend::VECTOR_TYPE gm1_;
            double valk_;
            double valkm1_;
            double diff_;
            double dphi_0_;
        };
    }

    namespace utils{

    template<class FUN>
    void check_grad(FUN const & fun, backend::VECTOR_TYPE const & x0){
        unsigned int dim = x0.size();
        backend::VECTOR_TYPE x(x0);
        backend::VECTOR_TYPE fgrad(dim);
        backend::VECTOR_TYPE numgrad(dim);
        double eps = 1e-8;
        fun(x,&fgrad);
        for(unsigned int i=0 ; i < dim ; ++i){
            double old = x(i);
            x(i) = old-eps; double vleft = fun(x,NULL);
            x(i) = old+eps; double vright = fun(x,NULL);
            numgrad(i) = (vright-vleft)/(2*eps);
        }
        std::cout << numgrad - fgrad << std::endl;
    }

    }



}
#endif
