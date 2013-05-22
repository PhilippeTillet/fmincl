/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_LINE_SEARCH_PHI_FUN_HPP_
#define FMINCL_LINE_SEARCH_PHI_FUN_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "fmincl/utils.hpp"
#include "interpolate.hpp"

namespace fmincl{

    namespace line_search{

        template<class Fun>
        class phi_fun{
        public:
            phi_fun(Fun const & fun, detail::state_ref const & state) : fun_(fun), state_(state), x_(state_.x.size()), g_(state_.x.size()){

            }
            double operator()(double alpha, double * dphi) {
                if(alpha != alpha_){
                    alpha_ = alpha;
                    x_ = state_.x + alpha_*state_.p;
                }
                if(dphi){
                    double res = fun_(x_,&g_);
                    *dphi = viennacl::linalg::inner_prod(g_,state_.p);
                    return res;
                }
                return fun_(x_, NULL);
            }
        private:
            Fun const & fun_;
            detail::state_ref const & state_;
            double alpha_;
            viennacl::vector<double> x_;
            viennacl::vector<double> g_;
        };

    }

}

#endif
