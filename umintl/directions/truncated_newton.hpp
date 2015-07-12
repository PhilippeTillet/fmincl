/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_
#define UMINTL_DIRECTIONS_TRUNCATED_NEWTON_HPP_

#include <vector>
#include <cmath>

#include "isaac/array.h"
#include "umintl/linear/conjugate_gradient.hpp"
#include "umintl/tools/shared_ptr.hpp"
#include "forwards.h"



namespace umintl{

  namespace tag{

    namespace truncated_newton{

      enum stopping_criterion{
        STOP_RESIDUAL_TOLERANCE,
        STOP_HV_VARIANCE
      };

    }

  }



  struct truncated_newton : public direction{
  private:
    
    

    struct compute_Ab: public linear::conjugate_gradient_detail::compute_Ab
    {
      compute_Ab(isaac::array const & x, isaac::array const & g, model_base const & model, umintl::detail::function_wrapper & fun) : x_(x), g_(g), model_(model), fun_(fun){ }
      virtual void operator()(isaac::array const & b, isaac::array & res){
        fun_.compute_hv_product(x_,g_,b,res,model_.get_hv_product_tag());
      }
    protected:
      isaac::array const & x_;
      isaac::array const & g_;
      model_base const & model_;
      umintl::detail::function_wrapper & fun_;
    };

    struct variance_stop_criterion : public linear::conjugate_gradient_detail::stopping_criterion
    {
    public:
      variance_stop_criterion(optimization_context & c) : c_(c){
        psi_=0;
      }

      void init(isaac::array const & p0){
        std::size_t H = c_.model().get_hv_product_tag().sample_size;
        std::size_t offset = c_.model().get_hv_product_tag().offset;
        isaac::array var(c_.N(), c_.dtype());
        c_.fun().compute_hv_product_variance(c_.x(),p0,var,hv_product_variance(STOCHASTIC,H,offset));
        double nrm2p0 = isaac::value_scalar(norm(p0, 2));
        double nrm1var = isaac::value_scalar(norm(var, 1));
        gamma_ = nrm1var/(H*std::pow(nrm2p0,2));
      }

      void update(isaac::array const & dk){
        psi_ = isaac::value_scalar(gamma_*isaac::pow(isaac::norm(dk),2));
      }

      bool operator()(double rsn){
        return rsn <= psi_;
      }

    private:
      optimization_context & c_;
      double psi_;
      double gamma_;
    };

  public:
    truncated_newton(tag::truncated_newton::stopping_criterion _stop = tag::truncated_newton::STOP_RESIDUAL_TOLERANCE, std::size_t _max_iter = 0) : max_iter(_max_iter), stop(_stop){ }

    virtual std::string info() const
    { return "Truncated Newton"; }

    void operator()(optimization_context & c)
    {
      if(max_iter==0) max_iter = c.N();

      linear::conjugate_gradient solver(max_iter, new compute_Ab(c.x(), c.g(),c.model(),c.fun()));
      if(stop==tag::truncated_newton::STOP_RESIDUAL_TOLERANCE)
      {
        double nrm2g = isaac::value_scalar(norm(c.g(),2));
        double tol = std::min(0.5,sqrt(nrm2g))*nrm2g;
        solver.stop = new linear::conjugate_gradient_detail::residual_norm(tol);
      }
      else
      {
        solver.stop = new variance_stop_criterion(c);
      }

      c.p()*=c.alpha();
      isaac::array b = - c.g();
      linear::conjugate_gradient::optimization_result res = solver(c.N(), c.p(), b, c.p());
      if(res.i==0 && res.ret == umintl::linear::conjugate_gradient::FAILURE_NON_POSITIVE_DEFINITE)
        c.p() = b;
      //std::cout << res.ret << " " << res.i << std::endl;
    }

    std::size_t max_iter;
    tag::truncated_newton::stopping_criterion stop;
  };

}

#endif
