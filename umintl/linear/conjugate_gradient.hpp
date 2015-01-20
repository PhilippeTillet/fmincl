/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_
#define UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_

#include <limits>
#include <cmath>

#include "atidlas/array.h"
#include "umintl/tools/shared_ptr.hpp"

namespace umintl{

  namespace linear{

    namespace conjugate_gradient_detail{

    /** @brief Base class for a stopping criterion for the linear conjugate gradient */
      struct stopping_criterion
      {
        public:
          virtual ~stopping_criterion(){ }
          virtual void init(atidlas::array const & p0) = 0;
          virtual void update(atidlas::array const & dk) = 0;
          virtual bool operator()(double rsn) = 0;
      };

      /** @brief residual norm stopping criterion
      *
      *  Stops the Linear CG when the norm of the residual is below a threshold
      */
      struct residual_norm : public stopping_criterion
      {
        public:
          residual_norm(double eps = 1e-4) : eps_(eps){ }
          void init(atidlas::array const & ){ }
          void update(atidlas::array const & ){ }
          bool operator()(double rsn){ return std::sqrt(rsn) < eps_; }
        private:
          double eps_;
      };

      /** @brief Base class for a matrix-vector product computation within linear conjugate gradient
      *
      * For the CG procedure, the explicit knowledge of the matrix is unnecessary. It is only necessary to know how to compute the product between
      * this matrix and any vector, hence this class
      */
      
      struct compute_Ab{
          virtual ~compute_Ab(){ }
          virtual void operator()(atidlas::array const & b, atidlas::array & res) = 0;
      };

      /** @brief symv product class */
      
      struct symv : public compute_Ab{
        public:
          symv(atidlas::array const & A) : A_(A){ }
          void operator()(atidlas::array const & b, atidlas::array & res)
          { res = atidlas::dot(A_, b); }
        private:
          atidlas::array const & A_;
      };


    }

    /** @brief Base class for the linear conjugate gradient
    *
    * This is a slightly modified version of the CG algorithm. Indeed,
    * the procedure is stopped whenever a direction of neative curvature is found
    */
    
    struct conjugate_gradient{
      private:
        
        
        
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

        optimization_result clear_terminate(return_code ret, std::size_t i)
        {
          optimization_result res;
          res.ret = ret;
          res.i = i;
          return res;
        }

      public:

        conjugate_gradient(std::size_t _max_iter
                          , conjugate_gradient_detail::compute_Ab * _compute_Ab
                          , conjugate_gradient_detail::stopping_criterion * _stop = new umintl::linear::conjugate_gradient_detail::residual_norm)
          : max_iter(_max_iter), compute_Ab(_compute_Ab), stop(_stop){ }


        optimization_result operator()(std::size_t N, atidlas::array const & x0, atidlas::array const & b, atidlas::array & x)
        {
          atidlas::numeric_type dtype = x0.dtype();
          atidlas::array r(atidlas::zeros(N, 1, dtype));
          atidlas::array p(atidlas::zeros(N, 1, dtype));
          atidlas::array Ap(atidlas::zeros(N, 1, dtype));
          atidlas::array best_x(atidlas::zeros(N, 1, dtype));

          double nrm_b = atidlas::value_scalar(norm(b));
          double nrm_x0 = atidlas::value_scalar(norm(x0));
          double lambda = 0;
          x = x0;
          if(nrm_x0==0)
            r = b;
          else
          {
            (*compute_Ab)(x,r); //r = Ax
            r = b - r; //r = b - Ax
          }
          p = r;
          stop->init(p);

          double rso = atidlas::value_scalar(dot(r, r));

          for(std::size_t i = 0 ; i < max_iter ; ++i){
            (*compute_Ab)(p,Ap);
            Ap += lambda*nrm_b*b;
            double pAp = atidlas::value_scalar(dot(p, Ap));
            if(pAp<0)
            {
              x = best_x;
              return clear_terminate(FAILURE_NON_POSITIVE_DEFINITE,i);
            }
            else
              best_x = x;

            double alpha = rso/pAp; //alpha = rso/(p'*Ap)
            x = x + alpha*p;
            r = r - alpha*Ap;
            stop->update(x);

            //double quadval = -0.5*(atidlas::dot(x,r) + atidlas::dot(N,x,b)); //quadval = -0.5*(x'r + x'b);

            double rsn = atidlas::value_scalar(dot(r,r));
            if((*stop)(rsn))
              return clear_terminate(SUCCESS,i);
            p = r + rsn/rso*p;
            rso = rsn;
          }
          return clear_terminate(FAILURE,max_iter);
        }

        std::size_t max_iter;
        tools::shared_ptr<linear::conjugate_gradient_detail::compute_Ab > compute_Ab;
        tools::shared_ptr<linear::conjugate_gradient_detail::stopping_criterion > stop;
    };

  }

}

#endif
