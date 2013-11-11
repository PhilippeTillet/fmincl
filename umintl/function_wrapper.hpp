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

#include "umintl/forwards.h"
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
            virtual void compute_value_gradient(VectorType const & x, ScalarType & value, VectorType & gradient) = 0;
            virtual void compute_hv_product(VectorType const & x, VectorType const & g, VectorType const & v, VectorType & Hv) = 0;
            virtual ~function_wrapper(){ }
        };


        template<class BackendType, class Fun>
        class function_wrapper_impl : public function_wrapper<BackendType>{
        private:
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::ScalarType ScalarType;
        private:

            //Compute both function's value and gradient
            void operator()(VectorType const &, ScalarType&, VectorType &, value_gradient const &, int2type<false>){
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
            void operator()(VectorType const & x, ScalarType& value, VectorType & gradient, value_gradient const & tag, int2type<true>){
                fun_(x,value,gradient,tag);
                n_value_computations_++;
                n_gradient_computations_++;
            }

            //Compute hessian-vector product
            void operator()(VectorType const &, VectorType const &, VectorType&, hessian_vector_product const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the hessian-vector product!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const & X, VectorType& v, VectorType & Hv, umintl::hessian_vector_product_tag)\n."
                            "If you wish to use right/centered-differentiation of the function's gradient, please set the appropriate options."
                            );
            }
            void operator()(VectorType const & x, VectorType const & v, VectorType& Hv, hessian_vector_product const & tag, int2type<true>){
                fun_(x,v,Hv,tag);
                n_hessian_vector_product_computations_++;
            }

        public:
            function_wrapper_impl(Fun & fun, std::size_t N, evaluation_policy_type const & evaluation_policy) : fun_(fun), N_(N), evaluation_policy_(evaluation_policy){
              n_value_computations_ = 0;
              n_gradient_computations_ = 0;
              n_hessian_vector_product_computations_ = 0;
            }

            unsigned int n_value_computations() const{ return n_value_computations_; }
            unsigned int n_gradient_computations() const { return n_gradient_computations_; }
            unsigned int n_hessian_vector_product_computations() const { return n_hessian_vector_product_computations_; }

            void compute_value_gradient(VectorType const & x, ScalarType & value, VectorType & gradient){
              (*this)(x,value,gradient,evaluation_policy_.model->get_value_gradient_tag(),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient)>::value>());
            }

            void compute_hv_product(VectorType const & x, VectorType const & g, VectorType const & v, VectorType & Hv){
              switch(evaluation_policy_.hessian_vector_product_computation){
                case umintl::CENTERED_DIFFERENCE:
                {
                  ScalarType dummy;
                  VectorType tmp = BackendType::create_vector(N_);
                  VectorType Hvleft = BackendType::create_vector(N_);
                  ScalarType h = 1e-7;

                  //Hv = Grad(x+hb)
                  BackendType::copy(N_,x,tmp); //tmp = x + hb
                  BackendType::axpy(N_,h,v,tmp);
                   (*this)(tmp,dummy,Hv,evaluation_policy_.model->get_value_gradient_tag(),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient)>::value>());

                  //Hvleft = Grad(x-hb)
                  BackendType::copy(N_,x,tmp); //tmp = x - hb
                  BackendType::axpy(N_,-h,v,tmp);
                  (*this)(tmp,dummy,Hvleft,evaluation_policy_.model->get_value_gradient_tag(),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient)>::value>());

                  //Hv-=Hvleft
                  //Hv/=2h
                  BackendType::axpy(N_,-1,Hvleft,Hv);
                  BackendType::scale(N_,1/(2*h),Hv);

                  BackendType::delete_if_dynamically_allocated(tmp);
                  BackendType::delete_if_dynamically_allocated(Hvleft);
                  break;
                }
                case umintl::FORWARD_DIFFERENCE:
                {
                  ScalarType dummy;
                  VectorType tmp = BackendType::create_vector(N_);
                  ScalarType h = 1e-7;

                  BackendType::copy(N_,x,tmp); //tmp = x + hb
                  BackendType::axpy(N_,h,v,tmp);
                  (*this)(tmp,dummy,Hv,evaluation_policy_.model->get_value_gradient_tag(),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient)>::value>());
                  BackendType::axpy(N_,-1,g,Hv);
                  BackendType::scale(N_,1/h,Hv);

                  BackendType::delete_if_dynamically_allocated(tmp);
                  break;
                }
                case umintl::PROVIDED:
                {
                  (*this)(x,v,Hv,evaluation_policy_.model->get_hv_product_tag(),int2type<is_call_possible<Fun,void(VectorType const &, VectorType&, VectorType&, hessian_vector_product)>::value>());
                  break;
                }
                default:
                  throw exceptions::incompatible_parameters("Unknown Hessian-Vector Product Computation Policy");
              }
            }
          private:
            Fun & fun_;
            std::size_t N_;
            evaluation_policy_type const & evaluation_policy_;

            unsigned int n_value_computations_;
            unsigned int n_gradient_computations_;
            unsigned int n_hessian_vector_product_computations_;
        };

    }

}
#endif
