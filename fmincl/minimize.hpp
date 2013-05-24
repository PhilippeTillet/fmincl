/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_MINIMIZE_HPP_
#define FMINCL_MINIMIZE_HPP_

#include <viennacl/vector.hpp>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include "fmincl/directions/cg.hpp"
#include "fmincl/directions/quasi-newton.hpp"
#include "fmincl/line_search/strong_wolfe_powell.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

    template<class Type>
    class proxy{
    public:
        template<class U>
        proxy & operator=(U const & u){ ptr = new U(u); }
        Type & get() const { return *ptr ; }
    private:
        Type * ptr;
    };

    struct optimization_options{
        proxy<detail::direction_base> direction;
        proxy<detail::line_search_base> line_search;
        unsigned int max_iter;
    };

    template<class Fun>
    viennacl::vector<double> minimize(Fun const & user_fun, viennacl::vector<double> const & x0, optimization_options const & options){
        detail::function_wrapper_impl<Fun> fun(user_fun);
        detail::state state(x0, fun);
        for( ; state.iter() < options.max_iter ; ++state.iter()){
            state.val() = state.fun()(state.x(), &state.g());
            if(state.iter()>0) std::cout << "iter " << state.iter() << " | cost : " << state.val() << std::endl;
            state.diff() = (state.val()-state.valm1());
            viennacl::backend::finish();
            options.direction.get()(state);
            state.dphi_0() = viennacl::linalg::inner_prod(state.p(),state.g());
            if(state.dphi_0()>0){
                state.p() = -state.g();
                state.dphi_0() = - viennacl::linalg::inner_prod(state.g(), state.g());
            }
            double ai = (state.iter()==0)?1:std::min(1.0d,1.01*2*state.diff()/state.dphi_0());
            std::pair<double, bool> search_res = options.line_search.get()(state, ai);
            if(search_res.second) break;
            state.x() = state.x() + search_res.first*state.p();
            state.valm1() = state.val();
        }
        return state.x();
    }

}

#endif
