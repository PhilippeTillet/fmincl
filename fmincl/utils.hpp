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

#include "tools/shared_ptr.hpp"
#include <iostream>

namespace fmincl{


    namespace detail{


        template<class BackendType>
        class function_wrapper{
        public:
            typedef typename BackendType::ScalarType ScalarType;
            typedef typename BackendType::VectorType VectorType;
            function_wrapper() : n_value_calc_(0), n_derivative_calc_(0){ }
            virtual void operator()(VectorType const & x, ScalarType * value, VectorType * grad) const = 0;
            unsigned int n_value_calc() const { return n_value_calc_; }
            unsigned int n_derivative_calc() const { return n_derivative_calc_; }
        protected:
            mutable unsigned int n_value_calc_;
            mutable unsigned int n_derivative_calc_;
        };

        template<class BackendType, class Fun>
        class function_wrapper_impl : public function_wrapper<BackendType>{
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::ScalarType ScalarType;
        public:
            function_wrapper_impl(Fun const & fun) : fun_(fun){ }
            void operator()(VectorType const & x, ScalarType * value, VectorType * grad) const {
                ++function_wrapper<BackendType>::n_value_calc_;
                if(grad) ++function_wrapper<BackendType>::n_derivative_calc_;
                return fun_(x, value, grad);
            }
        private:
            Fun const & fun_;
        };

        template<class BackendType>
        class optimization_context{
        private:
            optimization_context(optimization_context const & other);
            optimization_context& operator=(optimization_context const & other);
        public:
            typedef typename BackendType::ScalarType ScalarType;
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::MatrixType MatrixType;

            optimization_context(VectorType const & x0, std::size_t dim, detail::function_wrapper<BackendType> const & fun) : fun_(fun), iter_(0), dim_(dim){
                x_ = BackendType::create_vector(dim_);
                g_ = BackendType::create_vector(dim_);
                p_ = BackendType::create_vector(dim_);
                xm1_ = BackendType::create_vector(dim_);
                gm1_ = BackendType::create_vector(dim_);

                BackendType::copy(dim_,x0,x_);

                is_reinitializing_ = true;
            }

            detail::function_wrapper<BackendType> const & fun() { return fun_; }
            unsigned int & iter() { return iter_; }
            unsigned int & N() { return dim_; }
            VectorType & x() { return x_; }
            VectorType & g() { return g_; }
            VectorType & xm1() { return xm1_; }
            VectorType & gm1() { return gm1_; }
            VectorType & p() { return p_; }
            ScalarType & val() { return valk_; }
            ScalarType & valm1() { return valkm1_; }
            ScalarType & dphi_0() { return dphi_0_; }
            bool & is_reinitializing() { return is_reinitializing_; }

            ~optimization_context(){
                BackendType::delete_if_dynamically_allocated(x_);
                BackendType::delete_if_dynamically_allocated(g_);
                BackendType::delete_if_dynamically_allocated(p_);
                BackendType::delete_if_dynamically_allocated(xm1_);
                BackendType::delete_if_dynamically_allocated(gm1_);
            }

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
            ScalarType dphi_0_;

            bool is_reinitializing_;
        };
    }
}
#endif
