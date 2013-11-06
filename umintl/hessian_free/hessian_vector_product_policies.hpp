/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_HESSIAN_FREE_VECTOR_PRODUCT_HPP_
#define UMINTL_HESSIAN_FREE_VECTOR_PRODUCT_HPP_

#include "umintl/utils.hpp"
#include "umintl/linear/conjugate_gradient.hpp"
#include "umintl/linear/conjugate_gradient/compute_Ab/forwards.h"
#include <cmath>

namespace umintl{

  namespace hessian_free {

    template<class BackendType>
    struct hessian_vector_product_base : public linear::conjugate_gradient_detail::compute_Ab<BackendType>{
        virtual void init(optimization_context<BackendType> &){ }
        virtual void clean(optimization_context<BackendType> &){ }
    };

    template<class BackendType>
    struct hessian_vector_product_numerical_diff : public hessian_vector_product_base<BackendType>{
      private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::MatrixType MatrixType;
      public:
        hessian_vector_product_numerical_diff(ScalarType _h = 1e-7) : h(_h){ }
        void init(optimization_context<BackendType> & c){
          c_ = &c;
          tmp_ = BackendType::create_vector(c.N());
        }
        void clean(optimization_context<BackendType> &){
          c_ = NULL;
          BackendType::delete_if_dynamically_allocated(tmp_);
        }
        void operator()(std::size_t N, VectorType const & b, VectorType & res)
        {
          BackendType::copy(N,c_->x(),tmp_); //tmp = x + hb
          BackendType::axpy(N,h,b,tmp_);
          c_->fun()(tmp_,NULL,&res); // res = (Grad(tmp) - Grad(x))/h
          BackendType::axpy(N,-1,c_->g(),res);
          BackendType::scale(N,1/h,res);
        }
        ScalarType h;
      private:
        VectorType tmp_;
        optimization_context<BackendType>  * c_;
    };

    template<class BackendType, class Fun>
    struct hessian_vector_product_custom : public hessian_vector_product_base<BackendType>{
      private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::MatrixType MatrixType;
      public:
        hessian_vector_product_custom(Fun const & fun) : fun_(fun){ }
        void init(optimization_context<BackendType> & c){ c_ = &c; }
        void clean(optimization_context<BackendType> &){ c_ = NULL; }
        void operator()(std::size_t N, VectorType const & b, VectorType & res){
            fun_.compute_Hv(c_->x(), b, res);
        }

      private:
        Fun const & fun_;
        optimization_context<BackendType>  * c_;
    };

  }
}

#endif
