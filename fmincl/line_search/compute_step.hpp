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

        class strong_wolfe_powell {
        private:
            template<class PHI>
            std::pair<double, bool> zoom(double alo, double ahi, PHI & fun, detail::state_ref & state) const{
                viennacl::vector<double> xi(state.x.size());
                viennacl::vector<double> grad(state.x.size());
                double phi_alo, phi_ahi, dphi_alo, dphi_ahi;
                double aj, phi_aj, dphi_aj;
                while(1){
                    fun.set_alpha(alo); viennacl::backend::finish(); phi_alo = fun(&grad); dphi_alo = viennacl::linalg::inner_prod(grad,state.p);
                    fun.set_alpha(ahi); viennacl::backend::finish(); phi_ahi = fun(&grad); dphi_ahi = viennacl::linalg::inner_prod(grad,state.p);
                    if(alo < ahi)
                        aj = interpolator::cubicmin(alo, ahi, phi_alo, phi_ahi, dphi_alo, dphi_ahi);
                    else
                        aj = interpolator::cubicmin(ahi, alo, phi_ahi, phi_alo, dphi_ahi, dphi_alo);
                    if(aj==alo || aj==ahi){
                        return std::make_pair(ahi,true);
                    }
                    fun.set_alpha(aj); viennacl::backend::finish(); phi_aj = fun(NULL);
                    if(!sufficient_decrease(aj,phi_aj, state) || phi_aj >= phi_alo){
                        ahi = aj;
                    }
                    else{
                        phi_aj = fun(&grad); dphi_aj = viennacl::linalg::inner_prod(grad,state.p);
                        if(curvature(dphi_aj, state))
                            return std::make_pair(aj, false);
                        if(dphi_aj*(ahi - alo) >= 0)
                            ahi = alo;
                        alo = aj;
                    }
                }
            }

            bool sufficient_decrease(double ai, double phi_ai, detail::state_ref & state) const {
                return phi_ai <= (state.val + c1_*ai* state.dphi_0);
            }
            bool curvature(double dphi_ai, detail::state_ref & state) const{
                return std::abs(dphi_ai) <= c2_*std::abs(state.dphi_0);
            }

        public:            
            strong_wolfe_powell(double c1, double c2, double rho) :  c1_(c1), c2_(c2), rho_(rho){ }

            template<class PHI>
            std::pair<double, bool> operator()(PHI & fun, detail::state_ref & state) const{
                size_t dim = state.x.size();
                double aim1 = 0;
                double diff = state.val - state.valm1;
                double ai = (state.iter==0)?1:std::min(1.0d,1.01*2*diff/state.dphi_0);
                double phi_aim1 = state.val;
                double dphi_aim1 = state.dphi_0;
                double amax = 5;
                viennacl::vector<double> gi(dim);
                viennacl::vector<double> xi(dim);
                double phi_ai, dphi_ai;
                for(unsigned int i = 1 ; i<200; ++i){
                    fun.set_alpha(ai);
                    phi_ai = fun(NULL);

                    //Tests sufficient decrease
                    if(!sufficient_decrease(ai, phi_ai, state) || (i>1 && phi_ai >= phi_aim1))
                        return zoom(aim1, ai, fun, state);

                    fun(&gi);
                    dphi_ai = viennacl::linalg::inner_prod(gi,state.p);
                    //Tests curvature
                    if(curvature(dphi_ai, state))
                        return std::make_pair(ai, false);
                    if(dphi_ai>=0)
                        return zoom(ai, aim1, fun, state);

                    //Updates states
                    aim1 = ai;
                    phi_aim1 = phi_ai;
                    dphi_aim1 = dphi_ai;
                    ai = rho_*ai;
                    if(ai>amax)
                        return std::make_pair(amax,true);
                }
                return std::make_pair(amax,false);
            }
        private:
            double c1_;
            double c2_;
            double rho_;
        };

    }

}

#endif
