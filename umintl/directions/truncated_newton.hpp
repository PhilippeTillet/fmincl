/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_
#define UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "umintl/linear/conjugate_gradient.hpp"
#include "umintl/tools/shared_ptr.hpp"
#include "forwards.h"



namespace umintl{

template<class BackendType>
struct truncated_newton : public direction<BackendType>{
  private:
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;

    struct compute_Ab: public linear::conjugate_gradient_detail::compute_Ab<BackendType>{
        compute_Ab(VectorType const & x, VectorType const & g, umintl::detail::function_wrapper<BackendType> & fun) : x_(x), g_(g), fun_(fun){ }
        virtual void operator()(std::size_t, typename BackendType::VectorType const & b, typename BackendType::VectorType & res){
          fun_.compute_hv_product(x_,g_,b,res);
        }
      protected:
        VectorType const & x_;
        VectorType const & g_;
        umintl::detail::function_wrapper<BackendType> & fun_;
    };

  public:
    truncated_newton(std::size_t _max_iter = 0) : max_iter(_max_iter){ }

    void operator()(optimization_context<BackendType> & c){
      linear::conjugate_gradient<BackendType> solver(max_iter,new compute_Ab(c.x(), c.g(),c.fun()));
      if(max_iter==0)
          max_iter = c.N();
      ScalarType tol = std::min((ScalarType)0.5,std::sqrt(BackendType::nrm2(c.N(),c.g())));
      VectorType minus_g = BackendType::create_vector(c.N());
      BackendType::copy(c.N(),c.g(),minus_g);
      BackendType::scale(c.N(),-1,minus_g);
      BackendType::scale(c.N(),c.alpha(),c.p());
      typename linear::conjugate_gradient<BackendType>::optimization_result res = solver(c.N(),c.p(),minus_g,c.p(),tol);
      if(res.i==0 && res.ret == umintl::linear::conjugate_gradient<BackendType>::FAILURE_NON_POSITIVE_DEFINITE)
        BackendType::copy(c.N(),minus_g,c.p());
      //std::cout << res.ret << " " << res.i << std::endl;
      BackendType::delete_if_dynamically_allocated(minus_g);
    }

    std::size_t max_iter;
};

}

#endif
