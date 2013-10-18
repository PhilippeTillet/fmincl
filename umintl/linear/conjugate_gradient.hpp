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
          rr = BackendType::create_vector(N);
          p = BackendType::create_vector(N);
          Ap = BackendType::create_vector(N);
        }

        return_code clear_terminate(return_code ret){
          BackendType::delete_if_dynamically_allocated(r);
          BackendType::delete_if_dynamically_allocated(rr);
          BackendType::delete_if_dynamically_allocated(p);
          BackendType::delete_if_dynamically_allocated(Ap);
          return ret;
        }

      public:

        conjugate_gradient(std::size_t _max_iter, ScalarType _tolerance, conjugate_gradient_detail::compute_Ab<BackendType> * _compute_Ab = new umintl::linear::conjugate_gradient_detail::gemv<BackendType>()) :
          compute_Ab(_compute_Ab), max_iter(_max_iter), tolerance(_tolerance){ }

        return_code operator()(std::size_t N, MatrixType const & A, VectorType const & x0, VectorType const & b, VectorType & x)
        {
          allocate_tmp(N);

          //x = x0;
          BackendType::copy(N,x0,x);

          //r = Ax0 - b
          BackendType::symv(N,1,A,x0,0,r);
          BackendType::axpy(N,-1,b,r);

          //p = -r;
          BackendType::copy(N,r,p);
          BackendType::scale(N,-1,p);

          for(std::size_t i = 0 ; i < max_iter ; ++i){
            BackendType::symv(N,1,A,p,0,Ap); //Ap = A*p
            ScalarType alpha = BackendType::dot(N,r,r)/BackendType::dot(N,p,Ap); //alpha = r'*r/(p'*Ap)
            BackendType::axpy(N,alpha,p,x); //x = x + alpha*p
            BackendType::copy(N,r,rr); //rr = r + alpha*Ap
            BackendType::axpy(N,alpha,Ap,rr);
            ScalarType beta = BackendType::dot(N,rr,rr)/BackendType::dot(N,r,r); //beta = (rr'.rr)/(r'.r)
            BackendType::scale(N,beta,p);//pk = -rr + beta*pk
            BackendType::axpy(N,-1,rr,p);

            BackendType::copy(N,rr,r);//r = rr;

            ScalarType nrm_res = BackendType::nrm2(N,r);
            if(nrm_res < tolerance)
              return clear_terminate(SUCCESS);
          }
          return clear_terminate(FAILURE);
        }

        tools::shared_ptr<linear::conjugate_gradient_detail::compute_Ab<BackendType> > compute_Ab;
        std::size_t max_iter;
        ScalarType tolerance;

      private:
        VectorType r;
        VectorType rr;
        VectorType p;
        VectorType Ap;
    };

  }

}

#endif
