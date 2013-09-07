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

#include "fmincl/directions.hpp"
#include "fmincl/line_search.hpp"
#include "fmincl/stopping_criterion.hpp"


namespace fmincl{

    void fill_default_direction_line_search(optimization_options const & options){
      if(options.direction==NULL)
        options.direction = new quasi_newton_tag();
      if(options.line_search==NULL){
        if(dynamic_cast<quasi_newton_tag*>(options.direction.get()))
          options.line_search = new fmincl::strong_wolfe_powell_tag(1e-4,0.9);
        else
          options.line_search = new fmincl::strong_wolfe_powell_tag(1e-4,0.1);
      }
    }

    template<class BackendType>
    inline void print_context_infos(detail::optimization_context<BackendType> & context, optimization_options const & options){
        if(options.verbosity_level <2 )
            return;
        std::cout << "iter " << context.iter() << " | cost : " << context.val() << "| NVal : " << context.fun().n_value_calc() << std::endl;
    }


    template<class BackendType, class Fun>
    typename BackendType::VectorType minimize(Fun const & user_fun, typename BackendType::VectorType const & x0, optimization_options const & options){
        typedef typename BackendType::ScalarType ScalarType;

        fill_default_direction_line_search(options);
        detail::function_wrapper_impl<BackendType, Fun> fun(user_fun);
        detail::optimization_context<BackendType> context(x0, fun);
        context.val() = context.fun()(context.x(), &context.g());

        if(options.verbosity_level >= 1){
          std::cout << options.info();
        }

        tools::shared_ptr<direction_implementation<BackendType> > direction_impl(direction_mapping<BackendType>::type::create(*options.direction,context));
        tools::shared_ptr<line_search_implementation<BackendType> > line_search_impl(line_search_mapping<BackendType>::type::create(*options.line_search,context));
        tools::shared_ptr<stopping_criterion_implementation<BackendType> > stopping_criterion__impl(stopping_criterion_mapping<BackendType>::type::create(*options.stopping_criterion,context));
        for( ; context.iter() < options.max_iter ; ++context.iter()){
            print_context_infos(context,options);
            context.diff() = (context.val()-context.valm1());
            if(context.iter()==0){
              //Sets descent direction to gradient
              context.p() = -context.g();

              context.dphi_0() = BackendType::dot(context.p(),context.g());
            }
            else{
              //Update direction into context.p()
              (*direction_impl)();

              //Checks whether the direction is a descent direction or not
              context.dphi_0() = BackendType::dot(context.p(),context.g());
              if(context.dphi_0()>0){
                  context.p() = -context.g();
                  context.dphi_0() = - BackendType::dot(context.g(), context.g());
              }
            }

            double ai;
            if(context.iter()==0){
              ai = std::min(static_cast<ScalarType>(1.0),1/BackendType::asum(context.g()));
            }
            else{
              if(dynamic_cast<quasi_newton_implementation<BackendType> const *>(direction_impl.get()))
                ai = 1;
              else
                ai = std::min(1.0d,1.01*2*context.diff()/context.dphi_0());
            }

            //Perform line search to find the step size
            line_search_result<BackendType> search_res(context.dim());
            (*line_search_impl)(search_res, ai);

            if(search_res.has_failed || search_res.best_phi>context.val()) break;

            context.valm1() = context.val();
            context.xm1() = context.x();
            context.x() = search_res.best_x;
            context.val() = search_res.best_phi;
            context.gm1() = context.g();
            context.g() = search_res.best_g;

            if((*stopping_criterion__impl)())
              break;



        }
        return context.x();
    }

}

#endif
