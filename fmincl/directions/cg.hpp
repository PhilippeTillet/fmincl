/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CG_HPP_
#define FMINCL_DIRECTIONS_CG_HPP_

#include <viennacl/scalar.hpp>
#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>

#include "fmincl/utils.hpp"

namespace fmincl{

    namespace direction{


        struct polak_ribiere{
            viennacl::scalar<double> operator()(viennacl::vector<double> const & gk
                                                , viennacl::vector<double> const & gkm1){
                return viennacl::linalg::inner_prod(gk,  gk - gkm1)/viennacl::linalg::inner_prod(gkm1,gkm1);
            }
        };

        struct no_restart{
            bool operator()(){
                return false;
            }
        };


        template<class BETA_POLICY = polak_ribiere, class RESTART_POLICY = no_restart>
        class cg : public detail::direction_base{
            public:
              cg() { }
              void operator()(detail::state_ref & state){
                  if(gkm1_.empty() || restart())
                      state.p = -state.g;
                  else{
                    viennacl::scalar<double> beta = compute_beta(state.g, gkm1_);
                    state.p = -state.g + beta* state.p;
                  }
                  gkm1_ = state.g;
              }

            private:
              viennacl::vector<double> gkm1_;
              BETA_POLICY compute_beta;
              RESTART_POLICY restart;
        };

    }

}

#endif
