/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_LINE_SEARCH_HPP_
#define UMINTL_LINE_SEARCH_HPP_

#include "umintl/directions/conjugate_gradient.hpp"
#include "umintl/directions/steepest_descent.hpp"
#include "umintl/directions/quasi_newton.hpp"

#include "umintl/utils.hpp"
#include "forwards.h"

#include <cmath>

#include <map>

namespace umintl{

template<class BackendType>
struct strong_wolfe_powell : public line_search<BackendType>{
    //Tag
    strong_wolfe_powell(unsigned int _max_evals = 40) : line_search<BackendType>(_max_evals) { }

    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::MatrixType MatrixType;

    virtual void init(optimization_context<BackendType> & c){
        x0_ = BackendType::create_vector(c.N());
    }

    virtual void clean(optimization_context<BackendType> &){
        BackendType::delete_if_dynamically_allocated(x0_);
    }


private:
    ScalarType phi(int N, detail::function_wrapper<BackendType> const & fun, VectorType & x, VectorType const & x0, ScalarType alpha, VectorType const & p, VectorType & grad, ScalarType * dphi) const {
        ScalarType res = 0;
        //x = x0 + alpha*p;
        BackendType::copy(N,x0,x);
        BackendType::axpy(N,alpha,p,x);
        fun(x,&res, &grad);
        if(dphi){
            *dphi = BackendType::dot(N,grad,p);
        }
        return res;
    }

    bool sufficient_decrease(ScalarType ai, ScalarType phi_ai, ScalarType phi_0) const {
        return phi_ai <= (phi_0 + c1_*ai );
    }
    bool curvature(ScalarType dphi_ai, ScalarType dphi_0) const{
        return std::abs(dphi_ai) <= c2_*std::abs(dphi_0);
    }

    void zoom(line_search_result<BackendType> & res, ScalarType alo, ScalarType phi_alo, ScalarType dphi_alo, ScalarType ahi, ScalarType phi_ahi, ScalarType dphi_ahi, optimization_context<BackendType> & c, unsigned int max_evaluations) const{
        VectorType & current_x = res.best_x;
        VectorType & current_g = res.best_g;
        ScalarType & current_phi = res.best_phi;
        VectorType const & p = c.p();
        ScalarType eps = 1e-8;
        ScalarType aj = 0;
        ScalarType dphi_aj = 0;
        bool twice_close_to_boundary=false;
        for(unsigned int i = 0 ; i < max_evaluations ; ++i){
            ScalarType xmin = std::min(alo,ahi);
            ScalarType xmax = std::max(alo,ahi);
            if(alo < ahi)
                aj = cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi,xmin,xmax);
            else
                aj = cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo,xmin,xmax);
            if(std::min(xmax - aj, aj - xmin)/(xmax - xmin)  < eps){
                res.best_alpha = aj;
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
            current_phi = phi(c.N(), c.fun(), current_x, x0_, aj, p, current_g, &dphi_aj);
            if(!sufficient_decrease(aj,current_phi, c.val()) || current_phi >= phi_alo){
                ahi = aj;
                phi_ahi = current_phi;
                dphi_ahi = dphi_aj;

            }
            else{
                if(curvature(dphi_aj, c.dphi_0())){
                   res.best_alpha = aj;
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
        res.best_alpha = aj;
        res.has_failed=true;
    }

public:
    void operator()(line_search_result<BackendType> & res, umintl::direction<BackendType> * direction, optimization_context<BackendType> & c, ScalarType ai, unsigned int max_evaluations) {
        c1_ = 1e-4;
        if(dynamic_cast<quasi_newton<BackendType>*>(direction))
            c2_ = 0.9;
        else if(dynamic_cast<conjugate_gradient<BackendType>*>(direction))
            c2_ = 0.3;
        else
            c2_ = 0.9;

        ScalarType aim1 = 0;
        ScalarType phi_0 = c.val();
        ScalarType dphi_0 = c.dphi_0();
        ScalarType last_phi = phi_0;
        ScalarType dphi_aim1 = dphi_0;
        ScalarType dphi_ai;


        ScalarType & current_phi = res.best_phi;
        VectorType & current_x = res.best_x;
        VectorType & current_g = res.best_g;
        VectorType const & p = c.p();


        BackendType::copy(c.N(),c.x(), x0_);


        for(unsigned int i = 1 ; i< max_evaluations; ++i){
            current_phi = phi(c.N(),c.fun(), current_x, x0_, ai, p, current_g, &dphi_ai);

            //Tests sufficient decrease
            if(!sufficient_decrease(ai, current_phi, phi_0) || (i==1 && current_phi >= last_phi)){
                return zoom(res, aim1, last_phi, dphi_aim1, ai, current_phi, dphi_ai, c, max_evaluations-i);
            }

            //Tests curvature
            if(curvature(dphi_ai, dphi_0)){
                res.has_failed = false;
                res.best_alpha = ai;
                return;
            }
            if(dphi_ai>=0){
                return zoom(res, ai, current_phi, dphi_ai, aim1, last_phi, dphi_aim1, c, max_evaluations-i);
            }

            //Updates context_s
            ScalarType old_ai = ai;
            ScalarType old_phi_ai = current_phi;
            ScalarType old_dphi_ai = dphi_ai;

            //Cubic extrapolation to chose a new value of ai
            ScalarType xmin = ai + 0.01*(ai-aim1);
            ScalarType xmax = 10*ai;
            ai = cubicmin(aim1,ai,last_phi,current_phi,dphi_aim1,dphi_ai,xmin,xmax);
            if(std::abs(ai-xmin) < 1e-4 || std::abs(ai-xmax) < 1e-4)
                ai=(xmin+xmax)/2;
            aim1 = old_ai;
            last_phi = old_phi_ai;
            dphi_aim1 = old_dphi_ai;
        }
        res.best_alpha = ai;
        res.has_failed=true;
    }


private:
    ScalarType c1_;
    ScalarType c2_;
    ScalarType rho_;

    VectorType x0_;


};




}

#endif
