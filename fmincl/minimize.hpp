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

#include "fmincl/optimization_otions.hpp"

#include "fmincl/utils.hpp"

#include "fmincl/directions/conjugate_gradient.hpp"
#include "fmincl/directions/quasi_newton.hpp"

#include "fmincl/line_search/strong_wolfe_powell.hpp"

#include "fmincl/stopping_criterion/value_treshold.hpp"
#include "fmincl/stopping_criterion/gradient_treshold.hpp"


namespace fmincl{


    void fill_default_direction_line_search(optimization_options const & options){
      if(options.direction==NULL)
        options.direction = new quasi_newton();
      if(options.line_search==NULL){
        if(dynamic_cast<quasi_newton*>(options.direction.get()))
          options.line_search = new fmincl::strong_wolfe_powell(1e-4,0.9);
        else
          options.line_search = new fmincl::strong_wolfe_powell(1e-4,0.2);
      }
    }

    template<class BackendType>
    inline void print_context_infos(detail::optimization_context<BackendType> & context, optimization_options const & options){
        if(options.verbosity_level <2 )
            return;
        std::cout << "iter " << context.iter() << " | cost : " << context.val() << "| NVal : " << context.fun().n_value_calc() << std::endl;
    }


    template<class BackendType, class Fun>
    double minimize(typename BackendType::VectorType & res, Fun const & user_fun, typename BackendType::VectorType const & x0, std::size_t N, optimization_options const & options){
        typedef implementation_of<BackendType,direction,quasi_newton,conjugate_gradient> direction_mapping;
        typedef implementation_of<BackendType,line_search,strong_wolfe_powell> line_search_mapping;
        typedef implementation_of<BackendType,stopping_criterion,gradient_treshold,value_treshold> stopping_criterion_mapping;

        typedef typename BackendType::VectorType VectorType;

        fill_default_direction_line_search(options);
        detail::function_wrapper_impl<BackendType, Fun> fun(user_fun);
        detail::optimization_context<BackendType> state(x0, N, fun);
        state.val() = state.fun()(state.x(), &state.g());

        if(options.verbosity_level >= 1){
          std::cout << options.info();
        }

        tools::shared_ptr<direction::implementation<BackendType> > direction_impl(direction_mapping::create(*options.direction,state));
        tools::shared_ptr<line_search::implementation<BackendType> > line_search_impl(line_search_mapping::create(*options.line_search,state));
        tools::shared_ptr<stopping_criterion::implementation<BackendType> > stopping_criterion__impl(stopping_criterion_mapping::create(*options.stopping_criterion,state));

        double ai;
        line_search_result<BackendType> search_res(N);
        //double last_dphi_0;
        for( ; state.iter() < options.max_iter ; ++state.iter()){
            print_context_infos(state,options);

            if(state.iter()==0 || direction_impl->restart(state)){
              //Sets descent direction to gradient
              BackendType::copy(N,state.g(),state.p());
              BackendType::scale(N,-1,state.p());

              state.dphi_0() = BackendType::dot(N,state.p(),state.g());
              ai = std::min(static_cast<double>(1.0),1/BackendType::asum(N,state.g()));
            }
            else{
              //Update direction into context.p()
              (*direction_impl)(state);
              state.dphi_0() = BackendType::dot(N,state.p(),state.g());
              if(state.dphi_0()>0){
                  //Reset p = -g;
                  BackendType::copy(N,state.g(),state.p());
                  BackendType::scale(N,-1,state.p());

                  state.dphi_0() = - BackendType::dot(N,state.g(), state.g());
              }
              if(dynamic_cast<quasi_newton::implementation<BackendType> const *>(direction_impl.get()))
                ai = 1;
              else
                ai = std::min((double)1,2*(state.val() - state.valm1())/state.dphi_0());
            }

            (*line_search_impl)(search_res, state, ai);

            if(search_res.has_failed)
                break;

            BackendType::copy(N,state.x(),state.xm1());
            BackendType::copy(N,search_res.best_x,state.x());

            BackendType::copy(N,state.g(),state.gm1());
            BackendType::copy(N,search_res.best_g,state.g());

            state.valm1() = state.val();
            state.val() = search_res.best_phi;


            if((*stopping_criterion__impl)(state))
              break;
        }

        std::cout << state.iter() << std::endl;

        BackendType::copy(N,state.x(),res);
        return state.val();
    }

}

#endif
