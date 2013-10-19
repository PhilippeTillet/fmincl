/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_HESSIAN_FREE_SOLVER_HPP_
#define UMINTL_HESSIAN_FREE_SOLVER_HPP_

#include "umintl/utils.hpp"

#include "umintl/linear/conjugate_gradient.hpp"
#include "umintl/linear/conjugate_gradient/compute_Ab/forwards.h"

#include "hessian_vector_product_policies.hpp"

#include <cmath>

namespace umintl{

  namespace hessian_free {

    template<class BackendType>
    struct options : public linear::options<BackendType>{
      private:
        typedef typename BackendType::ScalarType ScalarType;
      public:
        options(std::size_t _max_iter, hessian_vector_product_base<BackendType> * _Hv_policy = new hessian_vector_product_numerical_diff<BackendType>()) : linear::options<BackendType>(_max_iter, _Hv_policy){ }
    };

    template<class BackendType>
    class solver
    {
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::ScalarType ScalarType;

      public:
        typedef typename linear::conjugate_gradient<BackendType>::optimization_result optimization_result;

        solver(hessian_free::options<BackendType> _options) : options(_options){ }

        void init(optimization_context<BackendType> & c){
          static_cast<hessian_vector_product_base<BackendType>*>(options.compute_Ab.get())->init(c);
          N_ = c.N();
        }

        void clean(optimization_context<BackendType> & c){
          static_cast<hessian_vector_product_base<BackendType>*>(options.compute_Ab.get())->clean(c);
        }

        optimization_result operator()(VectorType const & x0, VectorType const & b, VectorType & res, ScalarType tolerance){
          linear::conjugate_gradient<BackendType> solver(options);
          return solver(N_,x0,b,res,tolerance);
        }

        hessian_free::options<BackendType> options;
      private:
        std::size_t N_;
    };

  }

}

#endif
