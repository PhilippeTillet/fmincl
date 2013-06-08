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


#include "fmincl/backend.hpp"
#include "fmincl/directions.hpp"
#include "fmincl/line_search.hpp"
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
        unsigned int verbosity_level;
        unsigned int max_iter;
    };

    template<class Fun>
    backend::VECTOR_TYPE minimize(Fun const & user_fun, backend::VECTOR_TYPE const & x0, optimization_options const & options){
        detail::function_wrapper_impl<Fun> fun(user_fun);
        detail::state state(x0, fun);
        for( ; state.iter() < options.max_iter ; ++state.iter()){
            state.val() = state.fun()(state.x(), &state.g());
            utils::print_infos(options.verbosity_level, state);
            state.diff() = (state.val()-state.valm1());
            options.direction.get()(state);
            state.dphi_0() = backend::inner_prod(state.p(),state.g());
            if(state.dphi_0()>0){
                state.p() = -state.g();
                state.dphi_0() = - backend::inner_prod(state.g(), state.g());
            }
//            double ai = (state.iter()==0)?0.02:std::min(1.0d,1.01*2*state.diff()/state.dphi_0());
            double ai = (state.iter()==0)?std::min(1.0d,1/state.g().array().abs().sum()):1;
            std::pair<double, bool> search_res = options.line_search.get()(state, ai);
            if(search_res.second) break;
            state.x() = state.x() + search_res.first*state.p();
            state.valm1() = state.val();
        }
        return state.x();
    }

}

#endif
