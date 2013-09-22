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
#include "fmincl/directions/steepest_descent.hpp"

#include "fmincl/line_search/strong_wolfe_powell.hpp"

#include "fmincl/stopping_criterion/value_treshold.hpp"
#include "fmincl/stopping_criterion/gradient_treshold.hpp"


namespace fmincl{

    template<class BackendType>
    inline void print_context_infos(detail::optimization_context<BackendType> & context, optimization_options const & options){
        if(options.verbosity_level <2 )
            return;
        std::cout << "iter " << context.iter() << " | cost : " << context.val() << "| NVal : " << context.fun().n_value_calc() << std::endl;
    }


    template<class BackendType, class Fun>
    double minimize(typename BackendType::VectorType & res, Fun const & user_fun, typename BackendType::VectorType const & x0, std::size_t N, optimization_options const & options){
        typedef implementation_of<BackendType,direction,quasi_newton,conjugate_gradient,steepest_descent> direction_mapping;
        typedef implementation_of<BackendType,line_search,strong_wolfe_powell> line_search_mapping;
        typedef implementation_of<BackendType,stopping_criterion,gradient_treshold,value_treshold> stopping_criterion_mapping;

        typedef typename BackendType::VectorType VectorType;

        detail::function_wrapper_impl<BackendType, Fun> fun(user_fun);
        detail::optimization_context<BackendType> state(x0, N, fun);
        state.val() = state.fun()(state.x(), &state.g());

        if(options.verbosity_level >= 1){
          std::cout << options.info();
        }

        tools::shared_ptr<direction::implementation<BackendType> > fallback_direction(direction_mapping::create(fmincl::steepest_descent(),state));
        tools::shared_ptr<direction::implementation<BackendType> > default_direction(direction_mapping::create(*options.direction,state));
        tools::shared_ptr<direction::implementation<BackendType> > current_direction = default_direction;

        tools::shared_ptr<line_search::implementation<BackendType> > line_search(line_search_mapping::create(*options.line_search,state));
        tools::shared_ptr<stopping_criterion::implementation<BackendType> > stopping(stopping_criterion_mapping::create(*options.stopping_criterion,state));

        //double last_dphi_0;
        for( ; state.iter() < options.max_iter ; ++state.iter()){
            print_context_infos(state,options);
            current_direction = default_direction;
            if(state.iter()==0 || current_direction->restart(state)){
                current_direction = fallback_direction;
            }

            (*current_direction)(state);
            state.dphi_0() = BackendType::dot(N,state.p(),state.g());

            //Not a descent direction...
            if(state.dphi_0()>0){
                current_direction = fallback_direction;
                (*current_direction)(state);
                state.dphi_0() = BackendType::dot(N,state.p(),state.g());
             }

            line_search_result<BackendType> search_res(N);
            (*line_search)(search_res, current_direction.get(), state, current_direction->line_search_first_trial(state));

            if(search_res.has_failed)
                break;

            BackendType::copy(N,state.x(),state.xm1());
            BackendType::copy(N,search_res.best_x,state.x());

            BackendType::copy(N,state.g(),state.gm1());
            BackendType::copy(N,search_res.best_g,state.g());

            state.valm1() = state.val();
            state.val() = search_res.best_phi;


            if((*stopping)(state))
              break;
        }
        BackendType::copy(N,state.x(),res);
        return state.val();
    }

}

#endif
