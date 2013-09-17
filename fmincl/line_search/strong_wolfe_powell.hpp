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

#include "fmincl/utils.hpp"
#include "forwards.h"

#include <cmath>

namespace fmincl{

  struct strong_wolfe_powell : public line_search{
      strong_wolfe_powell(double _c1, double _c2) :  c1(_c1), c2(_c2) { }
      double c1;
      double c2;



      template<class BackendType>
      class implementation : public line_search::implementation<BackendType>{
        private:
          typedef typename line_search::implementation<BackendType>::VectorType VectorType;
          typedef typename line_search::implementation<BackendType>::MatrixType MatrixType;

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

          bool sufficient_decrease(double ai, double phi_ai, detail::optimization_context<BackendType> & context) const {
            return phi_ai <= (context.val() + c1_*ai );
          }
          bool curvature(double dphi_ai) const{
            return std::abs(dphi_ai) <= c2_*std::abs(context_.dphi_0());
          }

          void zoom(line_search_result<BackendType> & res, double alo, double phi_alo, double dphi_alo, double ahi, double phi_ahi, double dphi_ahi, detail::optimization_context<BackendType> & context) const{
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            double & current_phi = res.best_phi;
            VectorType const & p = context.p();
            double eps = 1e-6;
            double aj = 0;
            double dphi_aj = 0;

            bool twice_close_to_boundary = false;

            for(unsigned int i = 0 ; i < 10 ; ++i){
              double xmin = std::min(alo,ahi);
              double xmax = std::max(alo,ahi);
              if(alo < ahi)
                aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi,xmin,xmax);
              else
                aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo,xmin,xmax);
              if(std::min(xmax - aj, aj - xmin)/(xmax - xmin) < 0.1){
                  if(twice_close_to_boundary){
                      if( std::min(aj - xmin,xmax - aj) < eps){
                          res.has_failed=true;
                          return;
                      }
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
              current_phi = phi(N_, context.fun(), current_x, x0_, aj, p, current_g, &dphi_aj);
              if(!sufficient_decrease(aj,current_phi, context) || current_phi >= phi_alo){
                ahi = aj;
                phi_ahi = current_phi;
                dphi_ahi = dphi_aj;
              }
              else{
                if(curvature(dphi_aj)){
                    context.ak() = aj;
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
            res.has_failed = true;
          }



        public:
          implementation(strong_wolfe_powell const & tag, detail::optimization_context<BackendType> & context) : context_(context), N_(context.dim()),  c1_(tag.c1), c2_(tag.c2) {
              x0_ = BackendType::create_vector(N_);
          }

          ~implementation(){
              BackendType::delete_if_dynamically_allocated(x0_);
          }

          void operator()(line_search_result<BackendType> & res, double ai) {
            double aim1 = 0;
            double last_phi = context_.val();
            double dphi_aim1 = context_.dphi_0();
            double dphi_ai;

            double & current_phi = res.best_phi;
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            VectorType const & p = context_.p();


            BackendType::copy(N_,context_.x(), x0_);


            for(unsigned int i = 1 ; i<10; ++i){
              current_phi = phi(N_,context_.fun(), current_x, x0_, ai, p, current_g, &dphi_ai);

              //Tests sufficient decrease
              if(!sufficient_decrease(ai, current_phi, context_) || (i>1 && current_phi >= last_phi)){
                 return zoom(res, aim1, last_phi, dphi_aim1, ai, current_phi, dphi_ai, context_);
              }

              //Tests curvature
              if(curvature(dphi_ai)){
                context_.ak() = ai;
                res.has_failed = false;
                return;
              }
              if(dphi_ai>=0){
                return zoom(res, ai, current_phi, dphi_ai, aim1, last_phi, dphi_aim1, context_);
              }

              //Updates context_s
              double old_ai = ai;
              double old_phi_ai = current_phi;
              double old_dphi_ai = dphi_ai;

              //Cubic extrapolation to chose a new value of ai
              double xmin = ai + 0.01*(ai-aim1);
              double xmax = 10*ai;
              ai = cubicmin(aim1,ai,last_phi,current_phi,dphi_aim1,dphi_ai,xmin,xmax);

              aim1 = old_ai;
              last_phi = old_phi_ai;
              dphi_aim1 = old_dphi_ai;
            }
            res.has_failed=true;

          }
        private:
          detail::optimization_context<BackendType> & context_;
          int N_;

          double c1_;
          double c2_;
          double rho_;

          VectorType x0_;
      };
  };




}

#endif
