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

#include "fmincl/backend.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

  namespace detail{

    struct line_search_result{
      private:
        typedef backend::VECTOR_TYPE VEC;
      public:
        line_search_result(bool _has_failed,
                           double _best_f,
                           VEC const & _best_x,
                           VEC const & _best_g) : has_failed(_has_failed), best_f(_best_f), best_x(_best_x), best_g(_best_g){ }
        bool has_failed;
        double best_f;
        VEC best_x;
        VEC best_g;
    };

    class line_search_base{
      public:
        virtual line_search_result operator()(detail::state & state, double a_init) = 0;
    };

  }


  /* =========================== *
 * CUBIC INTERPOLATION
 * ===========================*/

  inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb, double xmin, double xmax){
    double d1 = dfa + dfb - 3*(fa - fb)/(a-b);
    double delta = pow(d1,2) - dfa*dfb;
    if(delta<0)
      return (xmin+xmax)/2;
    double d2 = std::sqrt(delta);
    double x = b - (b - a)*((dfb + d2 - d1)/(dfb - dfa + 2*d2));
    if(isnan(x))
      return (xmin+xmax)/2;
    return std::min(std::max(x,xmin+0.1*(xmax-xmin)),xmax-0.1*(xmax-xmin));
  }

  inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb){
    return cubicmin(a,b,fa,fb,dfa,dfb,std::min(a,b), std::max(a,b));
  }

  /* =========================== *
 * STRONG WOLFE POWELL
 * ===========================*/


  class strong_wolfe_powell : public detail::line_search_base {
    private:
      class phi_fun{
        public:
          void reset() { reset_ = true; }
          double operator()(detail::function_wrapper const & fun, backend::VECTOR_TYPE const & x, double alpha, backend::VECTOR_TYPE const & p, backend::VECTOR_TYPE & grad, double * dphi) {
            if(alpha != alpha_ || reset_){
              alpha_ = alpha;
              x_ = x + alpha_*p;
              reset_ = false;
            }
            double res = fun(x_,&grad);
            if(dphi){
              *dphi = backend::inner_prod(grad,p);
            }
            return res;
          }
          backend::VECTOR_TYPE const & x(){
            return x_;
          }

        private:
          bool reset_;
          double alpha_;
          backend::VECTOR_TYPE x_;
      };

      bool sufficient_decrease(double ai, double phi_ai, detail::state & state) const {
        return phi_ai <= (state.val() + c1_*ai   );
      }
      bool curvature(double dphi_ai, detail::state & state) const{
        return std::abs(dphi_ai) <= c2_*std::abs(state.dphi_0());
      }

      detail::line_search_result zoom(double alo, double ahi, detail::state & state) const{
        backend::VECTOR_TYPE const & x = state.x();
        backend::VECTOR_TYPE  & g = state.g();
        backend::VECTOR_TYPE const & p = state.p();
        double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
        double aj, phi_aj, dphi_aj;
        while(1){
          phi_alo = phi_(state.fun(), x, alo, p, g, &dphi_alo);
          phi_ahi = phi_(state.fun(), x, ahi, p, g, &dphi_ahi);
          if(alo < ahi)
            aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
          else
            aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
          if(aj==alo || aj==ahi){
            return detail::line_search_result(true,phi_ahi,phi_.x(),g);
          }
          phi_aj = phi_(state.fun(), x, aj, p, g, NULL);
          if(!sufficient_decrease(aj,phi_aj, state) || phi_aj >= phi_alo){
            ahi = aj;
          }
          else{
            phi_aj = phi_(state.fun(), x, aj, p, g, &dphi_aj);
            if(curvature(dphi_aj, state))
              return detail::line_search_result(false, phi_aj, phi_.x(), g);
            if(dphi_aj*(ahi - alo) >= 0)
              ahi = alo;
            alo = aj;
          }
        }
      }



    public:
      strong_wolfe_powell(double c1, double c2) :  c1_(c1), c2_(c2) { }

      detail::line_search_result operator()(detail::state & state, double ai) {
        phi_.reset();
        double aim1 = 0;
        double phi_aim1 = state.val();
        double dphi_aim1 = state.dphi_0();
        double phi_ai, dphi_ai;
        backend::VECTOR_TYPE const & x = state.x();
        backend::VECTOR_TYPE & g = state.g();
        backend::VECTOR_TYPE const & p = state.p();
        for(unsigned int i = 1 ; i<20; ++i){
          phi_ai = phi_(state.fun(), x, ai, p, g, &dphi_ai);

          //Tests sufficient decrease
          if(!sufficient_decrease(ai, phi_ai, state) || (i>1 && phi_ai >= phi_aim1))
            return zoom(aim1, ai, state);

          //Tests curvature
          if(curvature(dphi_ai, state))
            return detail::line_search_result(false,phi_ai,phi_.x(),g);
          if(dphi_ai>=0)
            return zoom(ai, aim1, state);

          //Updates states
          double old_ai = ai;
          double old_phi_ai = phi_ai;
          double old_dphi_ai = dphi_ai;

          //Cubic extrapolation to chose a new value of ai
          double xmin = ai + 0.01*(ai-aim1);
          double xmax = 10*ai;
          ai = cubicmin(aim1,ai,phi_aim1,phi_ai,dphi_aim1,dphi_ai,xmin,xmax);

          aim1 = old_ai;
          phi_aim1 = old_phi_ai;
          dphi_aim1 = old_dphi_ai;
        }
        return detail::line_search_result(true,phi_ai,phi_.x(),g);
      }
    private:
      double c1_;
      double c2_;
      double rho_;
      mutable phi_fun phi_; //phi is conceptually a const functor, but mutable because its temporary may not be always recalculated
  };


}

#endif
