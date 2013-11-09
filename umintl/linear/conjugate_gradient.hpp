/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_
#define UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_

#include "umintl/tools/shared_ptr.hpp"

#include <limits>

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
          FAILURE,
          FAILURE_NON_POSITIVE_DEFINITE
        };

        struct optimization_result{
            return_code ret;
            std::size_t i;
        };

      private:
        void allocate_tmp(std::size_t N){
          r = BackendType::create_vector(N);
          p = BackendType::create_vector(N);
          Ap = BackendType::create_vector(N);
        }

        optimization_result clear_terminate(return_code ret, std::size_t i){
          BackendType::delete_if_dynamically_allocated(r);
          BackendType::delete_if_dynamically_allocated(p);
          BackendType::delete_if_dynamically_allocated(Ap);
          optimization_result res;
          res.ret = ret;
          res.i = i;
          return res;
        }

      public:

        conjugate_gradient(std::size_t _max_iter, conjugate_gradient_detail::compute_Ab<BackendType> * _compute_Ab = new umintl::linear::conjugate_gradient_detail::symv<BackendType>())
            : max_iter(_max_iter), compute_Ab(_compute_Ab){ }

        conjugate_gradient(std::size_t _max_iter, tools::shared_ptr< conjugate_gradient_detail::compute_Ab<BackendType> > _compute_Ab)
            : max_iter(_max_iter), compute_Ab(_compute_Ab){ }


        optimization_result operator()(std::size_t N, VectorType const & x0, VectorType const & b, VectorType & x, ScalarType tolerance = 1e-4)
        {
          allocate_tmp(N);
          ScalarType nrm_b = BackendType::nrm2(N,b);

          //x = x0;
          BackendType::copy(N,x0,x);

          ScalarType nrm_x0 = BackendType::nrm2(N,x0);
          if(nrm_x0==0){
            //r = b
            BackendType::copy(N,b,r);
          }
          else{
            //r = b - Ax0
            (*compute_Ab)(N,x,r); //r = Ax
            BackendType::scale(N,-1,r); //r = -Ax
            BackendType::axpy(N,1,b,r); //r = b - Ax
          }

          //p = r;
          BackendType::copy(N,r,p);
          ScalarType rso = BackendType::dot(N,r,r);

          for(std::size_t i = 0 ; i < max_iter ; ++i){
            (*compute_Ab)(N,p,Ap);

             //Ap = A*p
            ScalarType pAp = BackendType::dot(N,p,Ap);

            if(pAp<1e-16){
              return clear_terminate(FAILURE_NON_POSITIVE_DEFINITE,i);
            }

            ScalarType alpha = rso/pAp; //alpha = rso/(p'*Ap)
            BackendType::axpy(N,alpha,p,x); //x = x + alpha*p
            BackendType::axpy(N,-alpha,Ap,r); //r = r - alpha*Ap

            ScalarType quadval = -0.5*(BackendType::dot(N,x,r) + BackendType::dot(N,x,b)); //quadval = -0.5*(x'r + x'b);

            ScalarType rsn = BackendType::dot(N,r,r);
            if(std::sqrt(rsn) < tolerance*nrm_b)
              return clear_terminate(SUCCESS,i);

            BackendType::scale(N,rsn/rso,p);//pk = r + rsn/rso*pk
            BackendType::axpy(N,1,r,p);
            rso = rsn;
          }
          return clear_terminate(FAILURE,std::numeric_limits<size_t>::max());
        }

        std::size_t max_iter;
        tools::shared_ptr<linear::conjugate_gradient_detail::compute_Ab<BackendType> > compute_Ab;
      private:
        VectorType r;
        VectorType p;
        VectorType Ap;
    };

  }

}

#endif
