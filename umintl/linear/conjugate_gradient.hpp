/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_
#define UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_

#include "umintl/tools/shared_ptr.hpp"

#include "conjugate_gradient/compute_Ab/forwards.h"
#include "conjugate_gradient/compute_Ab/gemv.hpp"

namespace umintl{

  namespace linear{

    template<class BackendType>
    struct options{
      private:
        typedef typename BackendType::ScalarType ScalarType;
      public:
        options(std::size_t _max_iter, ScalarType _tolerance, conjugate_gradient_detail::compute_Ab<BackendType> * _compute_Ab = new umintl::linear::conjugate_gradient_detail::symv<BackendType>()) : compute_Ab(_compute_Ab), max_iter(_max_iter), tolerance(_tolerance){ }

        tools::shared_ptr<linear::conjugate_gradient_detail::compute_Ab<BackendType> > compute_Ab;
        std::size_t max_iter;
        ScalarType tolerance;
    };

    template<class BackendType>
    struct conjugate_gradient{
      private:
        typedef typename BackendType::MatrixType MatrixType;
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::ScalarType ScalarType;
      public:
        enum return_code{
          SUCCESS,
          FAILURE
        };
      private:
        void allocate_tmp(std::size_t N){
          r = BackendType::create_vector(N);
          p = BackendType::create_vector(N);
          Ap = BackendType::create_vector(N);
        }

        return_code clear_terminate(return_code ret){
          BackendType::delete_if_dynamically_allocated(r);
          BackendType::delete_if_dynamically_allocated(p);
          BackendType::delete_if_dynamically_allocated(Ap);
          return ret;
        }

      public:

        conjugate_gradient(linear::options<BackendType> const & _options) : options(_options){ }

        return_code operator()(std::size_t N, VectorType const & x0, VectorType const & b, VectorType & x)
        {
          allocate_tmp(N);
          ScalarType nrm_b = BackendType::nrm2(N,b);

          //x = x0;
          BackendType::copy(N,x0,x);
          //r = b - Ax0
          (*options.compute_Ab)(N,x,r);
          BackendType::scale(N,-1,r);

          BackendType::axpy(N,1,b,r);
          //p = r;
          BackendType::copy(N,r,p);
          ScalarType rso = BackendType::dot(N,r,r);

          for(std::size_t i = 0 ; i < options.max_iter ; ++i){
            (*options.compute_Ab)(N,p,Ap);
             //Ap = A*p
            ScalarType alpha = rso/BackendType::dot(N,p,Ap); //alpha = rso/(p'*Ap)
            BackendType::axpy(N,alpha,p,x); //x = x + alpha*p
            BackendType::axpy(N,-alpha,Ap,r); //r = r - alpha*Ap
            ScalarType rsn = BackendType::dot(N,r,r);
            if(std::sqrt(rsn)/nrm_b < options.tolerance)
              return clear_terminate(SUCCESS);
            BackendType::scale(N,rsn/rso,p);//pk = r + rsn/rso*pk
            BackendType::axpy(N,1,r,p);
            rso = rsn;

          }
          return clear_terminate(FAILURE);
        }

        linear::options<BackendType> options;

      private:
        VectorType r;
        VectorType p;
        VectorType Ap;
    };

  }

}

#endif
