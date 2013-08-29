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

        template<class BackendType>
        class function_wrapper{
        public:
            typedef typename BackendType::VectorType VectorType;
            function_wrapper() : n_value_calc_(0), n_derivative_calc_(0){ }
            virtual double operator()(VectorType const & x, VectorType * grad) const = 0;
            unsigned int n_value_calc() const { return n_value_calc_; }
            unsigned int n_derivative_calc() const { return n_derivative_calc_; }
        protected:
            mutable unsigned int n_value_calc_;
            mutable unsigned int n_derivative_calc_;
        };

        template<class BackendType, class Fun>
        class function_wrapper_impl : public function_wrapper<BackendType>{
            typedef typename BackendType::VectorType VectorType;
        public:
            function_wrapper_impl(Fun const & fun) : fun_(fun){ }
            double operator()(VectorType const & x, VectorType * grad) const {
                ++function_wrapper<BackendType>::n_value_calc_;
                if(grad) ++function_wrapper<BackendType>::n_derivative_calc_;
                return fun_(x, grad);
            }
        private:
            Fun const & fun_;
        };

        template<class BackendType>
        class state{           
        public:
            typedef typename BackendType::ScalarType ScalarType;
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::MatrixType MatrixType;

            state(VectorType const & x0, detail::function_wrapper<BackendType> const & fun) : fun_(fun), iter_(0), dim_(x0.size()), x_(x0), g_(dim_), p_(dim_), xm1_(dim_), gm1_(dim_){ }

            detail::function_wrapper<BackendType> const & fun() { return fun_; }
            unsigned int & iter() { return iter_; }
            unsigned int & dim() { return dim_; }
            VectorType & x() { return x_; }
            VectorType & g() { return g_; }
            VectorType & xm1() { return xm1_; }
            VectorType & gm1() { return gm1_; }
            VectorType & p() { return p_; }
            ScalarType & val() { return valk_; }
            ScalarType & valm1() { return valkm1_; }
            ScalarType & diff() { return diff_; }
            ScalarType & dphi_0() { return dphi_0_; }
        private:
            detail::function_wrapper<BackendType> const & fun_;
            unsigned int iter_;
            unsigned int dim_;
            VectorType x_;
            VectorType g_;
            VectorType p_;
            VectorType xm1_;
            VectorType gm1_;
            ScalarType valk_;
            ScalarType valkm1_;
            ScalarType diff_;
            ScalarType dphi_0_;
        };
    }

    namespace utils{

    template<class FUN, class VectorType>
    void check_grad(FUN const & fun, VectorType const & x0){
        unsigned int dim = x0.size();
        VectorType x(x0);
        VectorType fgrad(dim);
        VectorType numgrad(dim);
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
