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

class line_search_base{
public:
    virtual std::pair<double, bool> operator()(detail::state & state, double a_init) = 0;
};

}


/* =========================== *
 * CUBIC INTERPOLATION
 * ===========================*/


inline double cubicmin(double a,double b, double fa, double fb, double dfa, double dfb){
    double eps = 1e-3;
    double bma = b - a;
    double fab = (fb - fa)/bma;
    double d1 = dfa + dfb - 3*fab;
    double delta = pow(d1,2) - dfa*dfb;
    if(delta>=0){
        double x;
        double d2 = std::sqrt(delta);
        double faab = (fab - dfa)/bma;
        double faabb = (dfb - 2*fab + dfa)/pow(bma,2);
        if(std::abs(faabb)<eps){
            if(std::abs(faab)<eps)
                x=a;
            else
                x= a - dfa/(2*faab);
        }
        else{
            x = b - bma*(dfb + d2 - d1)/(dfb - dfa + 2*d2);
        }
        x = std::max(a,std::min(x,b));
        double fx = (x-a)*(dfa + (x-a)*(faab + (x-b)*faabb));
        if(fa <= fb && fa <= fx)
            return a;
        if(fb <= fx)
            return b;
        return x;
    }
    if(fa <= fb)
        return a;
    return b;
}

/* =========================== *
 * STRONG WOLFE POWELL
 * ===========================*/


class strong_wolfe_powell : public detail::line_search_base {
private:
    class phi_fun{
    public:
        void reset() { reset_ = true; }
        double operator()(detail::function_wrapper const & fun, backend::VECTOR_TYPE const & x, double alpha, backend::VECTOR_TYPE const & p, double * dphi) {
            if(alpha != alpha_ || reset_){
                alpha_ = alpha;
                x_ = x + alpha_*p;
                reset_ = false;
            }
            if(dphi){
                backend::VECTOR_TYPE g(x.size());
                double res = fun(x_,&g);
                *dphi = backend::inner_prod(g,p);
                return res;
            }
            return fun(x_, NULL);
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

    std::pair<double, bool> zoom(double alo, double ahi, detail::state & state) const{
        backend::VECTOR_TYPE const & x = state.x();
        backend::VECTOR_TYPE const & p = state.p();
        double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
        double aj, phi_aj, dphi_aj;
        while(1){
            phi_alo = phi_(state.fun(), x, alo, p, &dphi_alo);
            phi_ahi = phi_(state.fun(), x, ahi, p, &dphi_ahi);
            if(alo < ahi)
                aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
            else
                aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
            if(aj==alo || aj==ahi){
                return std::make_pair(ahi,true);
            }
            phi_aj = phi_(state.fun(), x, aj, p, NULL);
            if(!sufficient_decrease(aj,phi_aj, state) || phi_aj >= phi_alo){
                ahi = aj;
            }
            else{
                phi_aj = phi_(state.fun(), x, aj, p, &dphi_aj);
                if(curvature(dphi_aj, state))
                    return std::make_pair(aj, false);
                if(dphi_aj*(ahi - alo) >= 0)
                    ahi = alo;
                alo = aj;
            }
        }
    }



public:
    strong_wolfe_powell(double c1, double c2, double rho) :  c1_(c1), c2_(c2), rho_(rho) { }

    std::pair<double, bool> operator()(detail::state & state, double ai) {
        phi_.reset();
        double aim1 = 0;
        double diff = state.val() - state.valm1();
        double phi_aim1 = state.val();
        double dphi_aim1 = state.dphi_0();
        double amax = 5;
        double phi_ai, dphi_ai;
        backend::VECTOR_TYPE const & x = state.x();
        backend::VECTOR_TYPE const & p = state.p();
        for(unsigned int i = 1 ; i<5; ++i){
            phi_ai = phi_(state.fun(), x, ai, p, &dphi_ai);

            //Tests sufficient decrease
            if(!sufficient_decrease(ai, phi_ai, state) || (i>1 && phi_ai >= phi_aim1))
                return zoom(aim1, ai, state);

            //Tests curvature
            if(curvature(dphi_ai, state))
                return std::make_pair(ai, false);
            if(dphi_ai>=0)
                return zoom(ai, aim1, state);

            //Updates states
            aim1 = ai;
            phi_aim1 = phi_ai;
            dphi_aim1 = dphi_ai;
            ai = rho_*ai;
            if(ai>amax)
                return std::make_pair(amax,true);
        }
        return std::make_pair(amax,true);
    }
private:
    double c1_;
    double c2_;
    double rho_;
    mutable phi_fun phi_; //phi is conceptually a const functor, but mutable because its temporary may not be always recalculated
};


}

#endif
