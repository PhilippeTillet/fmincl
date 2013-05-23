/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_
#define FMINCL_LINE_SEARCH_COMPUTE_STEP_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "fmincl/utils.hpp"
#include "interpolate.hpp"

namespace fmincl{

    namespace line_search{

        struct strong_wolfe_powell_tag{
            strong_wolfe_powell_tag(double _c1, double _c2, double _rho) : c1(_c1), c2(_c2), rho(_rho){ }
            double c1;
            double c2;
            double rho;
        };

        template<class FUN>
        class strong_wolfe_powell {
        private:
            class phi_fun{
            public:
                phi_fun(FUN const & fun) : fun_(fun){ }
                void reset() { reset_ = true; }
                double operator()(viennacl::vector<double> const & x, double alpha, viennacl::vector<double> const & p, double * dphi) {
                    if(alpha != alpha_ || reset_){
                        alpha_ = alpha;
                        x_ = x + alpha_*p;
                        reset_ = false;
                    }
                    if(dphi){
                        viennacl::vector<double> g(x.size());
                        double res = fun_(x_,&g);
                        *dphi = viennacl::linalg::inner_prod(g,p);
                        return res;
                    }
                    return fun_(x_, NULL);
                }
            private:
                FUN const & fun_;
                bool reset_;
                double alpha_;
                viennacl::vector<double> x_;
            };

            bool sufficient_decrease(double ai, double phi_ai, detail::state_ref & state) const {
                return phi_ai <= (state.val + params_.c1*ai* state.dphi_0);
            }
            bool curvature(double dphi_ai, detail::state_ref & state) const{
                return std::abs(dphi_ai) <= params_.c2*std::abs(state.dphi_0);
            }

            std::pair<double, bool> zoom(double alo, double ahi, detail::state_ref & state) const{
                viennacl::vector<double> const & x = state.x;
                viennacl::vector<double> const & p = state.p;
                double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
                double aj, phi_aj, dphi_aj;
                while(1){
                    phi_alo = phi_(x, alo, p, &dphi_alo);
                    phi_ahi = phi_(x, ahi, p, &dphi_ahi);
                    if(alo < ahi)
                        aj = interpolator::cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
                    else
                        aj = interpolator::cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
                    if(aj==alo || aj==ahi){
                        return std::make_pair(ahi,true);
                    }
                    phi_aj = phi_(x, aj, p, NULL);
                    if(!sufficient_decrease(aj,phi_aj, state) || phi_aj >= phi_alo){
                        ahi = aj;
                    }
                    else{
                        phi_aj = phi_(x, aj, p, &dphi_aj);
                        if(curvature(dphi_aj, state))
                            return std::make_pair(aj, false);
                        if(dphi_aj*(ahi - alo) >= 0)
                            ahi = alo;
                        alo = aj;
                    }
                }
            }



        public:            
            strong_wolfe_powell(FUN const & fun, strong_wolfe_powell_tag params) :  phi_(fun), params_(params) { }

            std::pair<double, bool> operator()(detail::state_ref & state) const{
                phi_.reset();
                double aim1 = 0;
                double diff = state.val - state.valm1;
                double ai = (state.iter==0)?1:std::min(1.0d,1.01*2*diff/state.dphi_0);
                double phi_aim1 = state.val;
                double dphi_aim1 = state.dphi_0;
                double amax = 5;
                double phi_ai, dphi_ai;
                viennacl::vector<double> const & x = state.x;
                viennacl::vector<double> const & p = state.p;
                for(unsigned int i = 1 ; i<20; ++i){
                    phi_ai = phi_(x, ai, p, NULL);

                    //Tests sufficient decrease
                    if(!sufficient_decrease(ai, phi_ai, state) || (i>1 && phi_ai >= phi_aim1))
                        return zoom(aim1, ai, state);

                    phi_(x, ai, p, &dphi_ai);

                    //Tests curvature
                    if(curvature(dphi_ai, state))
                        return std::make_pair(ai, false);
                    if(dphi_ai>=0)
                        return zoom(ai, aim1, state);

                    //Updates states
                    aim1 = ai;
                    phi_aim1 = phi_ai;
                    dphi_aim1 = dphi_ai;
                    ai = params_.rho*ai;
                    if(ai>amax)
                        return std::make_pair(amax,true);
                }
                return std::make_pair(amax,true);
            }
        private:
            strong_wolfe_powell_tag params_;
            //phi is conceptually a const functor, but mutable because its temporary may not be always recalculated
            mutable phi_fun phi_;

        };

    }

}

#endif
