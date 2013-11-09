/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_
#define UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "umintl/hessian-vector_product/forwards.h"
#include "umintl/hessian-vector_product/forward_difference.hpp"
#include "umintl/hessian-vector_product/provided_function.hpp"

#include "umintl/linear/conjugate_gradient.hpp"

#include "umintl/tools/shared_ptr.hpp"
#include "forwards.h"



namespace umintl{

template<class BackendType>
struct truncated_newton : public direction<BackendType>{
  private:
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
  public:
    truncated_newton(hessian_vector_product::base<BackendType> * _Hv_policy = new hessian_vector_product::forward_difference<BackendType>(), std::size_t _max_iter = 0) : Hv_policy(_Hv_policy), max_iter(_max_iter){ }

    void init(optimization_context<BackendType> & c){
      Hv_policy->init(c);
    }
    void clean(optimization_context<BackendType> & c){
      Hv_policy->clean(c);
    }

    virtual ScalarType line_search_first_trial(optimization_context<BackendType> &){
      return 1;
    }

    void operator()(optimization_context<BackendType> & c){
      linear::conjugate_gradient<BackendType> solver(max_iter,Hv_policy);
      if(max_iter==0)
          max_iter = c.N();
      ScalarType tol = std::min((ScalarType)0.5,(BackendType::nrm2(c.N(),c.g())));

      VectorType minus_g = BackendType::create_vector(c.N());
      BackendType::copy(c.N(),c.g(),minus_g);
      BackendType::scale(c.N(),-1,minus_g);
      typename linear::conjugate_gradient<BackendType>::optimization_result res = solver(c.N(),c.p(),minus_g,c.p(),tol);
      if(res.i==0 && res.ret == umintl::linear::conjugate_gradient<BackendType>::FAILURE_NON_POSITIVE_DEFINITE)
        BackendType::copy(c.N(),minus_g,c.p());
      std::cout << res.ret << " " << res.i << std::endl;
      BackendType::delete_if_dynamically_allocated(minus_g);
    }

    tools::shared_ptr< hessian_vector_product::base<BackendType> > Hv_policy;
    std::size_t max_iter;
};

}

#endif
