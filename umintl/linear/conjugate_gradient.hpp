/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_
#define UMINTL_DIRECTIONS_SECOND_ORDER_SOLVE_FORWARDS_H_

#include <limits>
#include <cmath>

#include "isaac/array.h"
#include "umintl/tools/shared_ptr.hpp"

namespace umintl{

  namespace linear{

    namespace conjugate_gradient_detail{

      /** @brief Base class for a stopping criterion for the linear conjugate gradient */
      struct stopping_criterion
      {
      public:
        virtual ~stopping_criterion(){ }
        virtual void init(isaac::array const & p0) = 0;
        virtual void update(isaac::array const & dk) = 0;
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
        void init(isaac::array const & ){ }
        void update(isaac::array const & ){ }
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
        virtual void operator()(isaac::array const & b, isaac::array & res) = 0;
      };

      /** @brief symv product class */
      
      struct symv : public compute_Ab{
      public:
        symv(isaac::array const & A) : A_(A){ }
        void operator()(isaac::array const & b, isaac::array & res)
        { res = isaac::dot(A_, b); }
      private:
        isaac::array const & A_;
      };


    }

    /** @brief Base class for the linear conjugate gradient
    *
    * This is a slightly modified version of the CG algorithm. Indeed,
    * the procedure is stopped whenever a direction of neative curvature is found
    */
    
    struct conjugate_gradient{
    public:
      enum return_code
      {
        SUCCESS,
        FAILURE,
        FAILURE_NON_POSITIVE_DEFINITE
      };

      struct optimization_result
      {
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


      optimization_result operator()(std::size_t N, isaac::array const & x0, isaac::array const & b, isaac::array & x)
      {
        isaac::numeric_type dtype = x0.dtype();
        isaac::array r(isaac::zeros(N, 1, dtype));
        isaac::array p(isaac::zeros(N, 1, dtype));
        isaac::array Ap(isaac::zeros(N, 1, dtype));
        isaac::array best_x(isaac::zeros(N, 1, dtype));

        double nrm_b = isaac::value_scalar(norm(b));
        double nrm_x0 = isaac::value_scalar(norm(x0));
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

        double rso = isaac::value_scalar(dot(r, r));

        for(std::size_t i = 0 ; i < max_iter ; ++i){
          (*compute_Ab)(p,Ap);
          Ap += lambda*nrm_b*b;
          double pAp = isaac::value_scalar(dot(p, Ap));
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

          //double quadval = -0.5*(isaac::dot(x,r) + isaac::dot(N,x,b)); //quadval = -0.5*(x'r + x'b);

          double rsn = isaac::value_scalar(dot(r,r));
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
