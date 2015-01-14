/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_FUNCTION_WRAPPER_HPP
#define UMINTL_FUNCTION_WRAPPER_HPP

#include "atidlas/array.h"
#include "tools/shared_ptr.hpp"
#include "tools/is_call_possible.hpp"
#include "tools/exception.hpp"

#include "umintl/forwards.h"


#include <iostream>



namespace umintl{

    namespace detail{
        
        template<int N>
        struct int2type{ };

        
        class function_wrapper{
            
            
        public:
            function_wrapper(){ }
            virtual unsigned int n_value_computations() const = 0;
            virtual unsigned int n_gradient_computations() const  = 0;
            virtual unsigned int n_hessian_vector_product_computations() const  = 0;
            virtual unsigned int n_datapoints_accessed() const = 0;
            virtual void compute_value_gradient(atidlas::array const & x, double & value, atidlas::array & gradient, value_gradient const & tag) = 0;
            virtual void compute_hv_product(atidlas::array const & x, atidlas::array const & g, atidlas::array const & v, atidlas::array & Hv, hessian_vector_product const & tag) = 0;
            virtual void compute_gradient_variance(atidlas::array const & x, atidlas::array & variance, gradient_variance const & tag) = 0;
            virtual void compute_hv_product_variance(atidlas::array const & x, atidlas::array const & v, atidlas::array & variance, hv_product_variance const & tag) = 0;
            virtual ~function_wrapper(){ }
        };


        template<class Fun>
        class function_wrapper_impl : public function_wrapper{
        private:
            //Compute gradient variance
            void operator()(atidlas::array const &, atidlas::array &, gradient_variance const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "Please provide an overload of :\n"
                            "void operator()(atidlas::array const & X, atidlas::array & variance, umintl::gradient_variance_tag)\n."
                            );
            }
            void operator()(atidlas::array const & x, atidlas::array & variance, gradient_variance const & tag, int2type<true>){
                fun_(x,variance,tag);
            }

            //Compute hessian_vector_product variance
            void operator()(atidlas::array const &, atidlas::array const & , atidlas::array &, hv_product_variance const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "Please provide an overload of :\n"
                            "void operator()(atidlas::array const & X, atidlas::array const & v, atidlas::array & variance, umintl::hessian_vector_product_variance_tag)\n."
                            );
            }
            void operator()(atidlas::array const & x, atidlas::array const & v, atidlas::array & variance, hv_product_variance const & tag, int2type<true>){
                fun_(x,v,variance,tag);
            }


            //Compute both function's value and gradient
            void operator()(atidlas::array const &, double&, atidlas::array &, value_gradient const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute both the function's value and gradient!"
                            "Please provide an overload of :\n"
                            "void operator()(atidlas::array const & X, double& value, atidlas::array & gradient, umintl::value_gradient_tag)\n."
                            "Alternatively, if you are computing the function's gradient and value separately,"
                            "check that minimizer.tweaks.function_gradient_evaluation is set to:"
                            "SEPARATE_FUNCTION_GRADIENT_EVALUATION."
                            );
            }
            void operator()(atidlas::array const & x, double& value, atidlas::array & gradient, value_gradient const & tag, int2type<true>){
                fun_(x,value,gradient,tag);
            }

            //Compute hessian-vector product
            void operator()(atidlas::array const &, atidlas::array const &, atidlas::array&, hessian_vector_product const &, int2type<false>){
                throw exceptions::incompatible_parameters(
                            "\n"
                            "No function supplied to compute the hessian-vector product!"
                            "Please provide an overload of :\n"
                            "void operator()(atidlas::array const & X, atidlas::array& v, atidlas::array & Hv, umintl::hessian_vector_product_tag)\n."
                            "If you wish to use right/centered-differentiation of the function's gradient, please set the appropriate options."
                            );
            }
            void operator()(atidlas::array const & x, atidlas::array const & v, atidlas::array& Hv, hessian_vector_product const & tag, int2type<true>){
                fun_(x,v,Hv,tag);
            }

        public:
            function_wrapper_impl(Fun & fun, std::size_t N, computation_type hessian_vector_product_computation) : fun_(fun), N_(N), hessian_vector_product_computation_(hessian_vector_product_computation){
              n_value_computations_ = 0;
              n_gradient_computations_ = 0;
              n_hessian_vector_product_computations_ = 0;
              n_datapoints_accessed_ = 0;
            }

            unsigned int n_datapoints_accessed() const{ return n_datapoints_accessed_; }
            unsigned int n_value_computations() const{ return n_value_computations_; }
            unsigned int n_gradient_computations() const { return n_gradient_computations_; }
            unsigned int n_hessian_vector_product_computations() const { return n_hessian_vector_product_computations_; }

            void compute_value_gradient(atidlas::array const & x,  double & value, atidlas::array & gradient, value_gradient const & tag){
              (*this)(x,value,gradient,tag,int2type<is_call_possible<Fun,void(atidlas::array const &, double&, atidlas::array&, value_gradient)>::value>());
              n_value_computations_++;
              n_gradient_computations_++;
              n_datapoints_accessed_+=tag.sample_size;
            }

            void compute_gradient_variance(atidlas::array const & x, atidlas::array & variance, gradient_variance const & tag){
              (*this)(x,variance,tag,int2type<is_call_possible<Fun,void(atidlas::array const &, atidlas::array &,gradient_variance)>::value>());
            }

            void compute_hv_product(atidlas::array const & x, atidlas::array const & g, atidlas::array const & v, atidlas::array & Hv, hessian_vector_product const & tag){
              value_gradient vgtag(tag.model, tag.sample_size, tag.offset);
              switch(hessian_vector_product_computation_){
                case umintl::CENTERED_DIFFERENCE:
                {
                  double dummy;
                  atidlas::array Hvleft(N_);
                  double h = 1e-7;

                  atidlas::array tmp = x + h*v;
                   (*this)(tmp,dummy,Hv,vgtag,int2type<is_call_possible<Fun,void(atidlas::array const &, double&, atidlas::array&, value_gradient)>::value>());

                  tmp = x - h*v;
                  (*this)(tmp,dummy,Hvleft,vgtag,int2type<is_call_possible<Fun,void(atidlas::array const &, double&, atidlas::array&, value_gradient)>::value>());

                  Hv = (Hv - Hvleft)/(2*h);
                  break;
                }
                case umintl::FORWARD_DIFFERENCE:
                {
                  double dummy;
                  double h = 1e-7;

                  atidlas::array tmp = x + h*v;
                  (*this)(tmp,dummy,Hv,vgtag,int2type<is_call_possible<Fun,void(atidlas::array const &, double&, atidlas::array&, value_gradient)>::value>());
                  Hv = (Hv - g)/h;
                  break;
                }
                case umintl::PROVIDED:
                {
                  (*this)(x,v,Hv,tag,int2type<is_call_possible<Fun,void(atidlas::array const &, atidlas::array&, atidlas::array&, hessian_vector_product)>::value>());
                  break;
                }
                default:
                  throw exceptions::incompatible_parameters("Unknown Hessian-Vector Product Computation Policy");
              }
              n_hessian_vector_product_computations_++;
              n_datapoints_accessed_+=tag.sample_size;
            }

            void compute_hv_product_variance(atidlas::array const & x, atidlas::array const & v, atidlas::array & variance, hv_product_variance const & tag){
              (*this)(x,v,variance,tag,int2type<is_call_possible<Fun,void(atidlas::array const &, atidlas::array const &, atidlas::array &,hv_product_variance)>::value>());
            }

          private:
            Fun & fun_;
            std::size_t N_;

            computation_type hessian_vector_product_computation_;

            unsigned int n_value_computations_;
            unsigned int n_gradient_computations_;
            unsigned int n_hessian_vector_product_computations_;

            unsigned int n_datapoints_accessed_;
        };

    }

}
#endif
