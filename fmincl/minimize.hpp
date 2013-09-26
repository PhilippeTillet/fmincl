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

#include "fmincl/optimization_options.hpp"
#include "fmincl/optimization_result.hpp"

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
        std::cout << "Ieration  " << context.iter() << " | cost : " << context.val() << "| NVal : " << context.fun().n_value_calc() << std::endl;
    }

    template<class BackendType>
    optimization_result terminate(optimization_result::termination_cause_type termination_cause, typename BackendType::VectorType & res, std::size_t N, detail::optimization_context<BackendType> & context){
        optimization_result result;
        BackendType::copy(N,context.x(),res);
        result.f = context.val();
        result.iteration = context.iter();
        result.n_functions_eval = context.fun().n_value_calc();
        result.n_gradient_eval = context.fun().n_derivative_calc();
        result.termination_cause = termination_cause;
        return result;
    }

    template<class BackendType, class Fun>
    optimization_result minimize(typename BackendType::VectorType & res, Fun const & user_fun, typename BackendType::VectorType const & x0, std::size_t N, optimization_options const & options){
        typedef implementation_of<BackendType,direction,quasi_newton,conjugate_gradient,steepest_descent> direction_mapping;
        typedef implementation_of<BackendType,line_search,strong_wolfe_powell> line_search_mapping;
        typedef implementation_of<BackendType,stopping_criterion,gradient_treshold,value_treshold> stopping_criterion_mapping;
        typedef typename BackendType::VectorType VectorType;

        detail::function_wrapper_impl<BackendType, Fun> fun(user_fun);
        detail::optimization_context<BackendType> c(x0, N, fun);

        tools::shared_ptr<direction::implementation<BackendType> > fallback_direction(direction_mapping::create(fmincl::steepest_descent(),c));
        tools::shared_ptr<direction::implementation<BackendType> > default_direction(direction_mapping::create(*options.direction,c));
        tools::shared_ptr<direction::implementation<BackendType> > current_direction = default_direction;

        tools::shared_ptr<line_search::implementation<BackendType> > line_search(line_search_mapping::create(*options.line_search,c));
        tools::shared_ptr<stopping_criterion::implementation<BackendType> > stopping(stopping_criterion_mapping::create(*options.stopping_criterion,c));

        line_search_result<BackendType> search_res(N);

        if(options.verbosity_level >= 1)
          std::cout << options.info();

        //First evaluation
        c.fun()(c.x(), &c.val(), &c.g());

        //Main loop
        for( ; c.iter() < options.max_iter ; ++c.iter()){
            print_context_infos(c,options);
//            for(std::size_t i = 0 ; i < N ; ++i){
//                std::cout << c.x()[i] << " " << std::flush;
//            }
//            std::cout << std::endl;
            current_direction = default_direction;
            if(c.is_reinitializing() || current_direction->restart(c)){
                current_direction = fallback_direction;
                c.is_reinitializing()=false;
            }

            (*current_direction)(c);
            c.dphi_0() = BackendType::dot(N,c.p(),c.g());

            //Not a descent direction...
            if(c.dphi_0()>0){
                //current_direction->reset(c);
                current_direction = fallback_direction;
                (*current_direction)(c);
                c.dphi_0() = BackendType::dot(N,c.p(),c.g());
             }

            (*line_search)(search_res, current_direction.get(), c, current_direction->line_search_first_trial(c));

            if(search_res.has_failed){
                return terminate(optimization_result::LINE_SEARCH_FAILED, res, N, c);
            }

            BackendType::copy(N,c.x(),c.xm1());
            BackendType::copy(N,search_res.best_x,c.x());

            BackendType::copy(N,c.g(),c.gm1());
            BackendType::copy(N,search_res.best_g,c.g());

            c.valm1() = c.val();
            c.val() = search_res.best_phi;


            if((*stopping)(c))
                return terminate(optimization_result::STOPPING_CRITERION, res, N, c);
        }

        return terminate(optimization_result::MAX_ITERATION_REACHED, res, N, c);
    }

}

#endif
