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
          typedef typename line_search::implementation<BackendType>::ScalarType ScalarType;
          typedef typename line_search::implementation<BackendType>::VectorType VectorType;
          typedef typename line_search::implementation<BackendType>::MatrixType MatrixType;

          ScalarType phi(int N, detail::function_wrapper<BackendType> const & fun, VectorType & x, VectorType const & x0, ScalarType alpha, VectorType const & p, VectorType & grad, ScalarType * dphi) const {

            //x = x0 + alpha*p;
            BackendType::copy(N,x0,x);
            BackendType::axpy(N,alpha,p,x);
            ScalarType res = fun(x,&grad);
            if(dphi){
              *dphi = BackendType::dot(N,grad,p);
            }
            return res;
          }

          bool sufficient_decrease(ScalarType ai, ScalarType phi_ai, detail::optimization_context<BackendType> & context) const {
            return phi_ai <= (context.val() + c1_*ai );
          }
          bool curvature(ScalarType dphi_ai) const{
            return std::abs(dphi_ai) <= c2_*std::abs(context_.dphi_0());
          }

          void zoom(line_search_result<BackendType> & res, ScalarType alo, ScalarType phi_alo, ScalarType dphi_alo, ScalarType ahi, ScalarType phi_ahi, ScalarType dphi_ahi, detail::optimization_context<BackendType> & context) const{
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            ScalarType & current_phi = res.best_phi;
            VectorType const & p = context.p();

            ScalarType eps = 1e-4;
            ScalarType aj = 0;
            ScalarType dphi_aj = 0;

            while(1){
              ScalarType xmin = std::min(alo,ahi);
              ScalarType xmax = std::max(alo,ahi);
              if(alo < ahi)
                aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi,xmin,xmax);
              else
                aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo,xmin,xmax);
              if( (aj - xmin)<eps || (xmax - aj) < eps){
                res.has_failed = true;
                return;
              }
              aj = std::min(std::max(aj,xmin+0.1f*(xmax-xmin)),xmax-0.1f*(xmax-xmin));
              current_phi = phi(N_, context.fun(), current_x, x0_, aj, p, current_g, &dphi_aj);
              if(!sufficient_decrease(aj,current_phi, context) || current_phi >= phi_alo){
                ahi = aj;
                phi_ahi = current_phi;
                dphi_ahi = dphi_aj;
              }
              else{
                if(curvature(dphi_aj)){
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

          }



        public:
          implementation(strong_wolfe_powell const & tag, detail::optimization_context<BackendType> & context) : context_(context), N_(context.dim()),  c1_(tag.c1), c2_(tag.c2) {
              x0_ = BackendType::create_vector(N_);
          }

          ~implementation(){
              BackendType::delete_if_dynamically_allocated(x0_);
          }

          void operator()(line_search_result<BackendType> & res, ScalarType ai) {
            ScalarType aim1 = 0;
            ScalarType last_phi = context_.val();
            ScalarType dphi_aim1 = context_.dphi_0();
            ScalarType dphi_ai;

            ScalarType & current_phi = res.best_phi;
            VectorType & current_x = res.best_x;
            VectorType & current_g = res.best_g;
            VectorType const & p = context_.p();


            BackendType::copy(N_,context_.x(), x0_);


            for(unsigned int i = 1 ; i<20; ++i){
              current_phi = phi(N_,context_.fun(), current_x, x0_, ai, p, current_g, &dphi_ai);

              //Tests sufficient decrease
              if(!sufficient_decrease(ai, current_phi, context_) || (i>1 && current_phi >= last_phi)){
                 return zoom(res, aim1, last_phi, dphi_aim1, ai, current_phi, dphi_ai, context_);
              }

              //Tests curvature
              if(curvature(dphi_ai)){
                res.has_failed = false;
                return;
              }
              if(dphi_ai>=0){
                return zoom(res, ai, current_phi, dphi_ai, aim1, last_phi, dphi_aim1, context_);
              }

              //Updates context_s
              ScalarType old_ai = ai;
              ScalarType old_phi_ai = current_phi;
              ScalarType old_dphi_ai = dphi_ai;

              //Cubic extrapolation to chose a new value of ai
              ScalarType xmin = ai + 0.01*(ai-aim1);
              ScalarType xmax = 10*ai;
              ai = cubicmin(aim1,ai,last_phi,current_phi,dphi_aim1,dphi_ai,xmin,xmax);

              aim1 = old_ai;
              last_phi = old_phi_ai;
              dphi_aim1 = old_dphi_ai;
            }
            res.has_failed = true;
          }
        private:
          detail::optimization_context<BackendType> & context_;
          int N_;

          ScalarType c1_;
          ScalarType c2_;
          ScalarType rho_;

          VectorType x0_;
      };
  };




}

#endif
