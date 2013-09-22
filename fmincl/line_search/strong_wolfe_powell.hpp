/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_LINE_SEARCH_HPP_
#define FMINCL_LINE_SEARCH_HPP_

#include "fmincl/directions/conjugate_gradient.hpp"
#include "fmincl/directions/steepest_descent.hpp"
#include "fmincl/directions/quasi_newton.hpp"

#include "fmincl/utils.hpp"
#include "forwards.h"

#include <cmath>

#include <map>

namespace fmincl{


  struct strong_wolfe_powell : public line_search{
      strong_wolfe_powell(){ }



      template<class BackendType>
      class implementation : public line_search::implementation<BackendType>{
        private:
          typedef typename BackendType::VectorType VectorType;
          typedef typename BackendType::MatrixType MatrixType;

          double phi(int N, detail::function_wrapper<BackendType> const & fun, VectorType & x, VectorType const & x0, double alpha, VectorType const & p, VectorType & grad, double * dphi) const {

            //x = x0 + alpha*p;
            BackendType::copy(N,x0,x);
            BackendType::axpy(N,alpha,p,x);
            double res = fun(x,&grad);
            if(dphi){
              *dphi = BackendType::dot(N,grad,p);
            }
            return res;
          }

          bool sufficient_decrease(double ai, double phi_ai, double phi_0) const {
            return phi_ai <= (phi_0 + c1_*ai );
          }
          bool curvature(double dphi_ai, double dphi_0) const{
            return std::abs(dphi_ai) <= c2_*std::abs(dphi_0);
          }

          void zoom(line_search_result<BackendType> & res, double alo, double phi_alo, double dphi_alo, double ahi, double phi_ahi, double dphi_ahi, detail::optimization_context<BackendType> & c) const{
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            double & current_phi = res.best_phi;
            VectorType const & p = c.p();
            double eps = 1e-16;
            double aj = 0;
            double dphi_aj = 0;
            bool twice_close_to_boundary=false;
            for(unsigned int i = 0 ; i < 40 ; ++i){
              double xmin = std::min(alo,ahi);
              double xmax = std::max(alo,ahi);
              if(alo < ahi)
                aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi,xmin,xmax);
              else
                aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo,xmin,xmax);
              if(std::min(xmax - aj, aj - xmin)/(xmax - xmin)  < eps){
                  res.has_failed=true;
                  return;
              }
              if(std::min(xmax - aj, aj - xmin)/(xmax - xmin) < 0.1){
                  if(twice_close_to_boundary){
                      if(std::abs(aj - xmax) < std::abs(aj - xmin))
                          aj = xmax - 0.1*(xmax-xmin);
                      else
                          aj = xmin + 0.1*(xmax-xmin);
                      twice_close_to_boundary = false;
                  }
                  else{
                      twice_close_to_boundary = true;
                  }
              }
              else{
                  twice_close_to_boundary = false;
              }
              current_phi = phi(N_, c.fun(), current_x, x0_, aj, p, current_g, &dphi_aj);
              if(!sufficient_decrease(aj,current_phi, c.val()) || current_phi >= phi_alo){
                ahi = aj;
                phi_ahi = current_phi;
                dphi_ahi = dphi_aj;

              }
              else{
                if(curvature(dphi_aj, c.dphi_0())){
                    res.has_failed = false;
                    return;
                }
                if(dphi_aj*(ahi - alo) >= 0){
                  ahi = alo;
                  phi_ahi = phi_alo;
                  dphi_ahi = dphi_alo;
                }
                alo = aj;
                phi_alo = current_phi;
                dphi_alo = dphi_aj;
              }
            }
            res.has_failed=true;
          }



        public:
          implementation(strong_wolfe_powell const &, detail::optimization_context<BackendType> & context) : N_(context.N()) {
              x0_ = BackendType::create_vector(N_);
          }

          ~implementation(){
              BackendType::delete_if_dynamically_allocated(x0_);
          }

          void operator()(line_search_result<BackendType> & res, fmincl::direction::implementation<BackendType> * direction, detail::optimization_context<BackendType> & c, double ai) {
            c1_ = 1e-4;
            if(dynamic_cast<quasi_newton::implementation<BackendType>*>(direction))
              c2_ = 0.9;
            else if(dynamic_cast<conjugate_gradient::implementation<BackendType>*>(direction))
              c2_ = 0.5;
            else
              c2_ = 0.9;

            double aim1 = 0;
            double phi_0 = c.val();
            double dphi_0 = c.dphi_0();
            double last_phi = phi_0;
            double dphi_aim1 = dphi_0;
            double dphi_ai;


            double & current_phi = res.best_phi;
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            VectorType const & p = c.p();


            BackendType::copy(N_,c.x(), x0_);


            for(unsigned int i = 1 ; i<40; ++i){
              current_phi = phi(N_,c.fun(), current_x, x0_, ai, p, current_g, &dphi_ai);

              //Tests sufficient decrease
              if(!sufficient_decrease(ai, current_phi, phi_0) || (i==1 && current_phi >= last_phi)){
                 return zoom(res, aim1, last_phi, dphi_aim1, ai, current_phi, dphi_ai, c);
              }

              //Tests curvature
              if(curvature(dphi_ai, dphi_0)){
                res.has_failed = false;
                return;
              }
              if(dphi_ai>=0){
                return zoom(res, ai, current_phi, dphi_ai, aim1, last_phi, dphi_aim1, c);
              }

              //Updates context_s
              double old_ai = ai;
              double old_phi_ai = current_phi;
              double old_dphi_ai = dphi_ai;

              //Cubic extrapolation to chose a new value of ai
              double xmin = ai + 0.01*(ai-aim1);
              double xmax = 10*ai;
              ai = cubicmin(aim1,ai,last_phi,current_phi,dphi_aim1,dphi_ai,xmin,xmax);
              if(std::abs(ai-xmin) < 1e-4 || std::abs(ai-xmax) < 1e-4)
                  ai=(xmin+xmax)/2;
              aim1 = old_ai;
              last_phi = old_phi_ai;
              dphi_aim1 = old_dphi_ai;
            }

            res.has_failed=true;
          }
        private:
          int N_;

          double c1_;
          double c2_;
          double rho_;

          VectorType x0_;
      };
  };




}

#endif
