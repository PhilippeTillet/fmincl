/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_
#define UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "umintl/hessian_free/hessian_vector_product_policies.hpp"
#include "umintl/hessian_free/solver.hpp"
#include "umintl/tools/shared_ptr.hpp"
#include "forwards.h"



namespace umintl{

template<class BackendType>
struct truncated_newton : public direction<BackendType>{
  private:
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
  public:
    truncated_newton(hessian_free::options<BackendType> const & _options) : solver_(_options){ }

    void init(optimization_context<BackendType> & c){
      solver_.init(c);
    }
    void clean(optimization_context<BackendType> & c){
      solver_.clean(c);
    }

    virtual ScalarType line_search_first_trial(optimization_context<BackendType> & c){
      return 1;
    }

    void operator()(optimization_context<BackendType> & c){
      ScalarType tol = std::min(0.5,std::sqrt(BackendType::nrm2(c.N(),c.g())));
      typename hessian_free::solver<BackendType>::optimization_result res = solver_(c.p(),c.g(),c.p(),tol);
      if(res.i==0 && res.ret == umintl::linear::conjugate_gradient<BackendType>::FAILURE_NON_POSITIVE_DEFINITE)
        BackendType::copy(c.N(),c.g(),c.p());
      BackendType::scale(c.N(),-1,c.p());
    }
  private:
    hessian_free::solver<BackendType> solver_;
};

}

#endif
