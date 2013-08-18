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
#include "fmincl/optimization_otions.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

    void fill_default_direction_line_search(optimization_options const & options){
      if(options.direction==NULL)
        options.direction = new quasi_newton();
      if(options.line_search==NULL){
        if(dynamic_cast<quasi_newton*>(options.direction.get()))
          options.line_search = new fmincl::strong_wolfe_powell(1e-4,0.9);
        else
          options.line_search = new fmincl::strong_wolfe_powell(1e-4,0.1);
      }
    }

    inline void print_state_infos(detail::state & state, optimization_options const & options){
        if(options.verbosity_level <2 )
            return;
        std::cout << "iter " << state.iter() << " | cost : " << state.val() << "| NVal : " << state.fun().n_value_calc() << std::endl;
    }

    template<class Fun>
    backend::VECTOR_TYPE minimize(Fun const & user_fun, backend::VECTOR_TYPE const & x0, optimization_options const & options){
        fill_default_direction_line_search(options);
        detail::function_wrapper_impl<Fun> fun(user_fun);
        detail::state state(x0, fun);
        state.val() = state.fun()(state.x(), &state.g());

        if(options.verbosity_level >= 1){
          std::cout << options.info();
        }

        for( ; state.iter() < options.max_iter ; ++state.iter()){
            print_state_infos(state,options);
            state.diff() = (state.val()-state.valm1());
            if(state.iter()==0){
              state.p() = -state.g();
              state.dphi_0() = backend::inner_prod(state.p(),state.g());
            }
            else{
              (*options.direction.get())(state);
              state.dphi_0() = backend::inner_prod(state.p(),state.g());
              if(state.dphi_0()>0){
                  state.p() = -state.g();
                  state.dphi_0() = - backend::inner_prod(state.g(), state.g());
              }
            }

            double ai;
            if(state.iter()==0){
              ai = std::min(1.0d,1/backend::abs_sum(state.g()));
            }
            else{
              if(dynamic_cast<quasi_newton*>(options.direction.get()))
                ai = 1;
              else
                ai = std::min(1.0d,1.01*2*state.diff()/state.dphi_0());
            }
            detail::line_search_result search_res = (*options.line_search.get())(state, ai);

            if(search_res.best_f>state.val()) break;

            state.valm1() = state.val();
            state.xm1() = state.x();
            state.x() = search_res.best_x;
            state.val() = search_res.best_f;
            state.gm1() = state.g();
            state.g() = search_res.best_g;

            if(std::abs(state.val() - state.valm1()) < 1e-6)
              break;



        }
        return state.x();
    }

}

#endif
