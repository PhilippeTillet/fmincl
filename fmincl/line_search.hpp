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
#include "fmincl/mapping.hpp"
#include "fmincl/tools/typelist.hpp"

#include <cmath>

namespace fmincl{

  template<class BackendType>
  struct line_search_result{
    private:
      typedef typename BackendType::VectorType VectorType;
      typedef typename BackendType::ScalarType ScalarType;
    public:
      line_search_result(bool _has_failed,
                         ScalarType _best_f,
                         VectorType const & _best_x,
                         VectorType const & _best_g) : has_failed(_has_failed), best_f(_best_f), best_x(_best_x), best_g(_best_g){ }
      bool has_failed;
      ScalarType best_f;
      VectorType best_x;
      VectorType best_g;
  };

  struct line_search_tag{ virtual ~line_search_tag(){ } };
  template<class BackendType>
  struct line_search_implementation{
  protected:
      typedef typename BackendType::ScalarType ScalarType;
      typedef typename BackendType::VectorType VectorType;
      typedef typename BackendType::MatrixType MatrixType;
  public:
      virtual line_search_result<BackendType> operator()(detail::state<BackendType> & state, ScalarType a_init) = 0;
  };

  /* =========================== *
 * CUBIC INTERPOLATION
 * ===========================*/

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb, ScalarType xmin, ScalarType xmax){
    ScalarType d1 = dfa + dfb - 3*(fa - fb)/(a-b);
    ScalarType delta = pow(d1,2) - dfa*dfb;
    if(delta<0)
      return (xmin+xmax)/2;
    ScalarType d2 = std::sqrt(delta);
    ScalarType x = b - (b - a)*((dfb + d2 - d1)/(dfb - dfa + 2*d2));
    if(isnan(x))
      return (xmin+xmax)/2;
    return std::min(std::max(x,xmin),xmax);
  }

  template<class ScalarType>
  inline ScalarType cubicmin(ScalarType a,ScalarType b, ScalarType fa, ScalarType fb, ScalarType dfa, ScalarType dfb){
    return cubicmin(a,b,fa,fb,dfa,dfb,std::min(a,b), std::max(a,b));
  }

  /* =========================== *
 * STRONG WOLFE POWELL
 * ===========================*/


  struct strong_wolfe_powell_tag : public line_search_tag{
      strong_wolfe_powell_tag(double _c1, double _c2) :  c1(_c1), c2(_c2) { }
      double c1;
      double c2;
  };

  template<class BackendType>
  class strong_wolfe_powell_implementation : public line_search_implementation<BackendType>{
    private:

      typedef typename line_search_implementation<BackendType>::ScalarType ScalarType;
      typedef typename line_search_implementation<BackendType>::VectorType VectorType;
      typedef typename line_search_implementation<BackendType>::MatrixType MatrixType;

      class phi_fun{
        public:
          ScalarType operator()(detail::function_wrapper<BackendType> const & fun, VectorType & x, VectorType const & x0, ScalarType alpha, VectorType const & p, VectorType & grad, ScalarType * dphi) {
            x = x0 + alpha*p;
            ScalarType res = fun(x,&grad);
            if(dphi){
              *dphi = BackendType::inner_prod(grad,p);
            }
            return res;
          }
      };

      bool sufficient_decrease(ScalarType ai, ScalarType phi_ai, detail::state<BackendType> & state) const {
        return phi_ai <= (state.val() + c1_*ai );
      }
      bool curvature(ScalarType dphi_ai, detail::state<BackendType> & state) const{
        return std::abs(dphi_ai) <= c2_*std::abs(state.dphi_0());
      }

      line_search_result<BackendType> zoom(ScalarType alo, ScalarType phi_alo, ScalarType dphi_alo, ScalarType ahi, ScalarType phi_ahi, ScalarType dphi_ahi, detail::state<BackendType> & state) const{
        unsigned int dim = state.dim();
        VectorType x0 = state.x();
        VectorType xj(dim);
        VectorType gj(dim);
        VectorType const & p = state.p();
        ScalarType eps = 1e-4;
        ScalarType aj = 0;
        ScalarType phi_aj = 0;
        ScalarType dphi_aj = 0;
        while(1){
          ScalarType xmin = std::min(alo,ahi);
          ScalarType xmax = std::max(alo,ahi);
          if(alo < ahi)
            aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi,xmin,xmax);
          else
            aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo,xmin,xmax);
          if( (aj - xmin)<eps || (xmax - aj) < eps)
            return line_search_result<BackendType>(true, phi_aj, xj, gj);
          aj = std::min(std::max(aj,xmin+0.1f*(xmax-xmin)),xmax-0.1f*(xmax-xmin));
          phi_aj = phi_(state.fun(), xj, x0, aj, p, gj, &dphi_aj);
          if(!sufficient_decrease(aj,phi_aj, state) || phi_aj >= phi_alo){
            ahi = aj;
            phi_ahi = phi_aj;
            dphi_ahi = dphi_aj;
          }
          else{
            if(curvature(dphi_aj, state))
              return line_search_result<BackendType>(false, phi_aj, xj, gj);
            if(dphi_aj*(ahi - alo) >= 0){
              ahi = alo;
              phi_ahi = phi_alo;
              dphi_ahi = dphi_alo;
            }
            alo = aj;
            phi_alo = phi_aj;
            dphi_alo = dphi_aj;
          }
        }
      }



    public:
      strong_wolfe_powell_implementation(strong_wolfe_powell_tag const & tag) :  c1_(tag.c1), c2_(tag.c2) { }

      line_search_result<BackendType> operator()(detail::state<BackendType> & state, ScalarType ai) {
        ScalarType aim1 = 0;
        ScalarType phi_aim1 = state.val();
        ScalarType dphi_aim1 = state.dphi_0();
        ScalarType phi_ai, dphi_ai;
        VectorType x = state.x();
        VectorType x0 = state.x();
        VectorType g = state.g();
        VectorType const & p = state.p();
        for(unsigned int i = 1 ; i<20; ++i){
          phi_ai = phi_(state.fun(), x, x0, ai, p, g, &dphi_ai);

          //Tests sufficient decrease
          if(!sufficient_decrease(ai, phi_ai, state) || (i>1 && phi_ai >= phi_aim1))
            return zoom(aim1, phi_aim1, dphi_aim1, ai, phi_ai, dphi_ai, state);

          //Tests curvature
          if(curvature(dphi_ai, state))
            return line_search_result<BackendType>(false,phi_ai,x,g);
          if(dphi_ai>=0)
            return zoom(ai, phi_ai, dphi_ai, aim1, phi_aim1, dphi_aim1, state);

          //Updates states
          ScalarType old_ai = ai;
          ScalarType old_phi_ai = phi_ai;
          ScalarType old_dphi_ai = dphi_ai;

          //Cubic extrapolation to chose a new value of ai
          ScalarType xmin = ai + 0.01*(ai-aim1);
          ScalarType xmax = 10*ai;
          ai = cubicmin(aim1,ai,phi_aim1,phi_ai,dphi_aim1,dphi_ai,xmin,xmax);

          aim1 = old_ai;
          phi_aim1 = old_phi_ai;
          dphi_aim1 = old_dphi_ai;
        }
        return line_search_result<BackendType>(true,phi_ai,x,g);
      }
    private:
      ScalarType c1_;
      ScalarType c2_;
      ScalarType rho_;
      mutable phi_fun phi_; //phi is conceptually a const functor, but mutable because its temporary may not be always recalculated
  };


  template<class BackendType>
  struct line_search_mapping{
      typedef implementation_from_tag<typename make_typelist<FMINCL_CREATE_MAPPING(strong_wolfe_powell)>::type
                                     ,line_search_tag,line_search_implementation<BackendType> > type;
  };


}

#endif
