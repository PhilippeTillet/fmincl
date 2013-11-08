/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_FUNCTION_WRAPPER_HPP
#define UMINTL_FUNCTION_WRAPPER_HPP

#include "tools/shared_ptr.hpp"
#include "tools/is_call_possible.hpp"
#include "tools/exception.hpp"
#include "tags.hpp"
#include <iostream>



namespace umintl{


    namespace detail{
        
        template<int N>
        struct int2type{ };

        template<class BackendType>
        class function_wrapper{
            typedef typename BackendType::ScalarType ScalarType;
            typedef typename BackendType::VectorType VectorType;
        public:
            function_wrapper(){ }
            virtual unsigned int n_value_computations() const = 0;
            virtual unsigned int n_gradient_computations() const  = 0;
            virtual unsigned int n_hessian_vector_product_computations() const  = 0;
            virtual void compute_value(VectorType const & x, ScalarType & value) = 0;
            virtual void compute_gradient(VectorType const & x, VectorType & gradient) = 0;
            virtual void compute_value_gradient(VectorType const & x, ScalarType & value, VectorType & grad) = 0;
            virtual void compute_hessian_vector_product(VectorType const & x, VectorType const & v, VectorType & Hv) = 0;
            virtual ~function_wrapper(){ }
        };


        template<class BackendType, class Fun>
        class function_wrapper_impl : public function_wrapper<BackendType>{
        private:
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::ScalarType ScalarType;
        private:

            //Compute function's value alone
            void compute_value(VectorType const &, ScalarType &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the function's value alone!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const &, ScalarType &, umintl::value_tag)\n."
                            "Alternatively, if you are computing the function's value along with the gradient,"
                            "check that minimizer.tweaks.function_gradient_evaluation is set to:"
                            "PACKED_FUNCTION_GRADIENT_EVALUATION."
                            );
            }
            void compute_value(VectorType const & x, ScalarType & value, int2type<true>){
                fun_(x,value,value_tag());
                n_value_computations_++;
            }

            //Compute function's gradient alone
            void compute_gradient(VectorType const &, VectorType &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the function's gradient alone!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const & X, VectorType & gradient, umintl::gradient_tag)\n."
                            "Alternatively, if you are computing the function's gradient along with the value,"
                            "check that minimizer.tweaks.function_gradient_evaluation is set to:"
                            "PACKED_FUNCTION_GRADIENT_EVALUATION."
                            );
            }
            void compute_gradient(VectorType const & x, VectorType & gradient, int2type<true>){
                fun_(x,gradient,gradient_tag());
                n_gradient_computations_++;
            }

            //Compute both function's value and gradient
            void compute_value_gradient(VectorType const &, ScalarType&, VectorType &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute both the function's value and gradient!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const & X, ScalarType& value, VectorType & gradient, umintl::value_gradient_tag)\n."
                            "Alternatively, if you are computing the function's gradient and value separately,"
                            "check that minimizer.tweaks.function_gradient_evaluation is set to:"
                            "SEPARATE_FUNCTION_GRADIENT_EVALUATION."
                            );
            }
            void compute_value_gradient(VectorType const & x, ScalarType& value, VectorType & gradient, int2type<true>){
                fun_(x,value,gradient,value_gradient_tag());
                n_value_computations_++;
                n_gradient_computations_++;
            }

            //Compute hessian-vector product
            void compute_hessian_vector_product(VectorType const &, VectorType const &, VectorType&, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the hessian-vector product!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const & X, VectorType& v, VectorType & Hv, umintl::hessian_vector_product_tag)\n."
                            "If you wish to use right/centered-differentiation of the function's gradient, please set the appropriate options."
                            );
            }
            void compute_hessian_vector_product(VectorType const & x, VectorType const & v, VectorType& Hv, int2type<true>){
                fun_(x,v,Hv,hessian_vector_product_tag());
                n_hessian_vector_product_computations_++;
            }

        public:
            function_wrapper_impl(Fun & fun) : fun_(fun){
                n_value_computations_ = 0;
                n_gradient_computations_ = 0;
                n_hessian_vector_product_computations_ = 0;
            }

            unsigned int n_value_computations() const{
                return n_value_computations_;
            }

            unsigned int n_gradient_computations() const {
                return n_gradient_computations_;
            }

            unsigned int n_hessian_vector_product_computations() const {
                return n_hessian_vector_product_computations_;
            }

            void compute_value(VectorType const & x, ScalarType & value){
                compute_value(x,value,int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, value_tag)>::value>());
            }

            void compute_gradient(VectorType const & x, VectorType & gradient){
                compute_gradient(x,gradient,int2type<is_call_possible<Fun,void(VectorType const &, VectorType&, gradient_tag)>::value>());
            }

            void compute_value_gradient(VectorType const & x, ScalarType & value, VectorType & gradient){
                compute_value_gradient(x,value,gradient,int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient_tag)>::value>());
            }

            void compute_hessian_vector_product(VectorType const & x, VectorType const & v, VectorType & Hv){
                compute_hessian_vector_product(x,v,Hv,int2type<is_call_possible<Fun,void(VectorType const &, VectorType&, VectorType&, hessian_vector_product_tag)>::value>());
            }
        private:
            Fun & fun_;

            unsigned int n_value_computations_;
            unsigned int n_gradient_computations_;
            unsigned int n_hessian_vector_product_computations_;
        };

    }

}
#endif
