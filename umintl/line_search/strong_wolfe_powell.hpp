/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINE_SEARCH_STRONG_WOLFE_POWELL_HPP_
#define UMINTL_LINE_SEARCH_STRONG_WOLFE_POWELL_HPP_

#include "umintl/directions/conjugate_gradient.hpp"
#include "umintl/directions/steepest_descent.hpp"
#include "umintl/directions/quasi_newton.hpp"
#include "umintl/directions/truncated_newton.hpp"


#include "umintl/optimization_context.hpp"
#include "forwards.h"

#include <cmath>

#include <map>

namespace umintl{

  /** @brief The strong wolfe-powell line-search class
 */

  struct strong_wolfe_powell : public line_search{
    //Tag
    /** @brief The constructor
     *  @param _max_evals maximum number of value-gradient evaluation in the line-search
     */
    strong_wolfe_powell(unsigned int _max_evals = 40) : line_search(_max_evals) { }

  private:
    using line_search::max_evals;

    /** @brief Sufficient decrease test for the strong wolfe-powell conditions */
    bool sufficient_decrease(double alpha, double phi_alpha, double phi0) const
    { return phi_alpha <= (phi0 + c1_*alpha ); }

    /** @brief Curvature test for the strong wolfe-powell conditions */
    bool curvature(double dphi_alpha, double dphi0) const
    { return std::abs(dphi_alpha) <= c2_*std::abs(dphi0); }

    void zoom(line_search_result & res, isaac::array const & x0, double alpha_low, double phi_alpha_low, double dphi_alpha_low
              , double alpha_high, double phi_alpha_high, double dphi_alpha_high
              , optimization_context & c, unsigned int eval_offset) const{
      isaac::array & current_x = res.best_x;
      isaac::array & current_g = res.best_g;
      double & current_phi = res.best_phi;
      isaac::array const & p = c.p();
      double eps = 1e-8;
      double alpha = 0;
      double dphi = 0;
      bool twice_close_to_boundary=false;
      for(unsigned int i = eval_offset ; i < max_evals ; ++i){
        double xmin = std::min(alpha_low,alpha_high);
        double xmax = std::max(alpha_low,alpha_high);
        if(alpha_low < alpha_high)
          alpha = cubicmin(alpha_low, alpha_high, phi_alpha_low, phi_alpha_high, dphi_alpha_low, dphi_alpha_high,xmin,xmax);
        else
          alpha = cubicmin(alpha_high, alpha_low, phi_alpha_high, phi_alpha_low, dphi_alpha_high, dphi_alpha_low,xmin,xmax);
        if(std::min(xmax - alpha, alpha - xmin)/(xmax - xmin)  < eps){
          res.best_alpha = alpha;
          res.has_failed=true;
          return;
        }
        if(std::min(xmax - alpha, alpha - xmin)/(xmax - xmin) < 0.1){
          if(twice_close_to_boundary){
            if(std::abs(alpha - xmax) < std::abs(alpha - xmin))
              alpha = xmax - 0.1*(xmax-xmin);
            else
              alpha = xmin + 0.1*(xmax-xmin);
            twice_close_to_boundary = false;
          }
          else{
            twice_close_to_boundary = true;
          }
        }
        else{
          twice_close_to_boundary = false;
        }

        //Compute phi(alpha) = f(x0 + alpha*p)
        current_x = x0 + alpha*p;
        c.fun().compute_value_gradient(current_x,current_phi,current_g,c.model().get_value_gradient_tag());
        dphi = isaac::value_scalar(dot(current_g, p));

        if(!sufficient_decrease(alpha,current_phi, c.val()) || current_phi >= phi_alpha_low){
          alpha_high = alpha;
          phi_alpha_high = current_phi;
          dphi_alpha_high = dphi;

        }
        else{
          if(curvature(dphi, c.dphi_0())){
            res.best_alpha = alpha;
            res.has_failed = false;
            return;
          }
          if(dphi*(alpha_high - alpha_low) >= 0){
            alpha_high = alpha_low;
            phi_alpha_high = phi_alpha_low;
            dphi_alpha_high = dphi_alpha_low;
          }
          alpha_low = alpha;
          phi_alpha_low = current_phi;
          dphi_alpha_low = dphi;
        }
      }
      res.best_alpha = alpha;
      res.has_failed=true;
    }

  public:

    /** @brief Line-Search procedure call
    *
    * @param res reference to line search result
    * @param direction the descent direction procedure used for the line search
    * @param c corresponding optimization context
    */
    void operator()(line_search_result & res, umintl::direction * direction, optimization_context & c) {
      double alpha;
      c1_ = 1e-4;
      if(dynamic_cast<conjugate_gradient* >(direction) || dynamic_cast<steepest_descent* >(direction)){
        c2_ = 0.2;
        alpha = isaac::value_scalar(minimum(isaac::value_scalar(1, c.dtype()), 1/sum(abs(c.g()))));
      }
      else{
        c2_ = 0.9;
        alpha = 1;
      }

      double alpham1 = 0;
      double phi_0 = c.val();
      double dphi_0 = c.dphi_0();
      double last_phi = phi_0;
      double dphim1 = dphi_0;
      double dphi;


      double & current_phi = res.best_phi;
      isaac::array & current_x = res.best_x;
      isaac::array & current_g = res.best_g;
      isaac::array const & p = c.p();

      isaac::array x0 = c.x();

      for(unsigned int i = 1 ; i< max_evals; ++i){
        //Compute phi(alpha) = f(x0 + alpha*p) ; dphi = grad(phi)_alpha'*p
        current_x = x0 + alpha*p;
        c.fun().compute_value_gradient(current_x,current_phi,current_g,c.model().get_value_gradient_tag());
        dphi = isaac::value_scalar(dot(current_g, p));

        //Tests sufficient decrease
        if(!sufficient_decrease(alpha, current_phi, phi_0) || (i==1 && current_phi >= last_phi)){
          return zoom(res, x0, alpham1, last_phi, dphim1, alpha, current_phi, dphi, c, i);
        }

        //Tests curvature
        if(curvature(dphi, dphi_0)){
          res.has_failed = false;
          res.best_alpha = alpha;
          return;
        }
        if(dphi>=0){
          return zoom(res, x0, alpha, current_phi, dphi, alpham1, last_phi, dphim1, c, i);
        }

        //Updates context_s
        double old_alpha = alpha;
        double old_phi = current_phi;
        double old_dphi = dphi;

        //Cubic extrapolation to chose a new value of ai
        double xmin = alpha + 0.01*(alpha-alpham1);
        double xmax = 10*alpha;
        alpha = cubicmin(alpham1,alpha,last_phi,current_phi,dphim1,dphi,xmin,xmax);
        if(std::abs(alpha-xmin) < 1e-4 || std::abs(alpha-xmax) < 1e-4)
          alpha=(xmin+xmax)/2;
        alpham1 = old_alpha;
        last_phi = old_phi;
        dphim1 = old_dphi;
      }
      res.best_alpha = alpha;
      res.has_failed=true;
    }


  private:
    /** parameter of the strong-wolfe powell conditions */
    double c1_;
    /** parameter of the strong-wolfe powell conditions */
    double c2_;
  };




}

#endif
