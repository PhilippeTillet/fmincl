/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_FUNCTION_WRAPPER_HPP
#define UMINTL_FUNCTION_WRAPPER_HPP

#include "umintl/evaluation_policies.hpp"

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
            virtual void compute_value(std::size_t i, VectorType const & x, ScalarType & value) = 0;
            virtual void compute_gradient(std::size_t i, VectorType const & x, VectorType & gradient) = 0;
            virtual void compute_value_gradient(std::size_t i, VectorType const & x, ScalarType & value, VectorType & gradient) = 0;
            virtual void compute_hv_product(std::size_t i, VectorType const & x, VectorType const & g, VectorType const & v, VectorType & Hv) = 0;
            virtual ~function_wrapper(){ }
        };


        template<class BackendType, class Fun>
        class function_wrapper_impl : public function_wrapper<BackendType>{
        private:
            typedef typename BackendType::VectorType VectorType;
            typedef typename BackendType::ScalarType ScalarType;
        private:

            //Compute function's value alone
            void operator()(VectorType const &, ScalarType &, value_tag const &, int2type<false>){
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
            void operator()(VectorType const & x, ScalarType & value, value_tag const & tag, int2type<true>){
                fun_(x,value,tag);
                n_value_computations_++;
            }

            //Compute function's gradient alone
            void operator()(VectorType const &, VectorType &, gradient_tag const &, int2type<false>){
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
            void operator()(VectorType const & x, VectorType & gradient, gradient_tag const & tag, int2type<true>){
                fun_(x,gradient,tag);
                n_gradient_computations_++;
            }

            //Compute both function's value and gradient
            void operator()(VectorType const &, ScalarType&, VectorType &, value_gradient_tag const &, int2type<false>){
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
            void operator()(VectorType const & x, ScalarType& value, VectorType & gradient, value_gradient_tag const & tag, int2type<true>){
                fun_(x,value,gradient,tag);
                n_value_computations_++;
                n_gradient_computations_++;
            }

            //Compute hessian-vector product
            void operator()(VectorType const &, VectorType const &, VectorType&, hessian_vector_product_tag const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the hessian-vector product!"
                            "Please provide an overload of :\n"
                            "void operator()(VectorType const & X, VectorType& v, VectorType & Hv, umintl::hessian_vector_product_tag)\n."
                            "If you wish to use right/centered-differentiation of the function's gradient, please set the appropriate options."
                            );
            }
            void operator()(VectorType const & x, VectorType const & v, VectorType& Hv, hessian_vector_product_tag const & tag, int2type<true>){
                fun_(x,v,Hv,tag);
                n_hessian_vector_product_computations_++;
            }

        public:
            function_wrapper_impl(Fun & fun, std::size_t N, evaluation_policies_type evaluation_policies) : fun_(fun), N_(N), evaluation_policies_(evaluation_policies){
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

            void compute_value(std::size_t i, VectorType const & x, ScalarType & value){
              model_type_base * model = evaluation_policies_.value.model.get();
              model->update(i);
              (*this)(x,value,value_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, value_tag)>::value>());
            }

            void compute_gradient(std::size_t i, VectorType const & x, VectorType & gradient){
              model_type_base * model = evaluation_policies_.gradient.model.get();
              model->update(i);
              (*this)(x,gradient,gradient_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, VectorType&, gradient_tag)>::value>());
            }

            void compute_value_gradient(std::size_t i, VectorType const & x, ScalarType & value, VectorType & gradient){
              model_type_base * model = evaluation_policies_.value_gradient.model.get();
              model->update(i);
              (*this)(x,value,gradient,value_gradient_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient_tag)>::value>());
            }

            void compute_hv_product(std::size_t i, VectorType const & x, VectorType const & g, VectorType const & v, VectorType & Hv){
              model_type_base * model = evaluation_policies_.hv_product.model.get();
              model->update(i);
              switch(evaluation_policies_.hv_product.computation){
                case hv_product_evaluation_policy::CENTERED_DIFFERENCE:
                {
                  ScalarType dummy;
                  VectorType tmp = BackendType::create_vector(N_);
                  VectorType Hvleft = BackendType::create_vector(N_);
                  ScalarType h = 1e-7;

                  //Hv = Grad(x+hb)
                  BackendType::copy(N_,x,tmp); //tmp = x + hb
                  BackendType::axpy(N_,h,v,tmp);
                   (*this)(tmp,dummy,Hv,value_gradient_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient_tag)>::value>());

                  //Hvleft = Grad(x-hb)
                  BackendType::copy(N_,x,tmp); //tmp = x - hb
                  BackendType::axpy(N_,-h,v,tmp);
                  (*this)(tmp,dummy,Hvleft,value_gradient_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient_tag)>::value>());

                  //Hv-=Hvleft
                  //Hv/=2h
                  BackendType::axpy(N_,-1,Hvleft,Hv);
                  BackendType::scale(N_,1/(2*h),Hv);

                  BackendType::delete_if_dynamically_allocated(tmp);
                  BackendType::delete_if_dynamically_allocated(Hvleft);
                  break;
                }
                case hv_product_evaluation_policy::FORWARD_DIFFERENCE:
                {
                  ScalarType dummy;
                  VectorType tmp = BackendType::create_vector(N_);
                  ScalarType h = 1e-7;

                  BackendType::copy(N_,x,tmp); //tmp = x + hb
                  BackendType::axpy(N_,h,v,tmp);
                  (*this)(tmp,dummy,Hv,value_gradient_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, ScalarType&, VectorType&, value_gradient_tag)>::value>());
                  BackendType::axpy(N_,-1,g,Hv);
                  BackendType::scale(N_,1/h,Hv);

                  BackendType::delete_if_dynamically_allocated(tmp);
                  break;
                }
                case hv_product_evaluation_policy::PROVIDED:
                {
                  (*this)(x,v,Hv,hessian_vector_product_tag(*model),int2type<is_call_possible<Fun,void(VectorType const &, VectorType&, VectorType&, hessian_vector_product_tag)>::value>());
                  break;
                }
                default:
                  throw exceptions::incompatible_parameters("Unknown Hessian-Vector Product Computation Policy");
              }
            }
          private:
            Fun & fun_;
            std::size_t N_;
            evaluation_policies_type evaluation_policies_;

            unsigned int n_value_computations_;
            unsigned int n_gradient_computations_;
            unsigned int n_hessian_vector_product_computations_;
        };

    }

}
#endif
