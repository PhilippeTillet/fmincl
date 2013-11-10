/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_MINIMIZE_HPP_
#define UMINTL_MINIMIZE_HPP_

#include "umintl/optimization_options.hpp"
#include "umintl/optimization_result.hpp"

#include "umintl/function_wrapper.hpp"
#include "umintl/optimization_context.hpp"

#include "umintl/directions/conjugate_gradient.hpp"
#include "umintl/directions/quasi_newton.hpp"
#include "umintl/directions/steepest_descent.hpp"
#include "umintl/directions/truncated_newton.hpp"

#include "umintl/line_search/strong_wolfe_powell.hpp"

#include "umintl/stopping_criterion/value_treshold.hpp"
#include "umintl/stopping_criterion/gradient_treshold.hpp"

#include "umintl/model_type/forwards.h"
#include "umintl/model_type/deterministic.hpp"
#include "umintl/model_type/stochastic.hpp"


namespace umintl{

    template<class BackendType>
    class minimizer{
    public:
        minimizer(umintl::direction<BackendType> * _direction = new quasi_newton<BackendType>()
                             , umintl::stopping_criterion<BackendType> * _stopping_criterion = new gradient_treshold<BackendType>()
                             , unsigned int iter = 1024, unsigned int verbosity = 0) :
            direction(_direction)
          , fallback_direction(new steepest_descent<BackendType>())
          , line_search(new strong_wolfe_powell<BackendType>())
          , stopping_criterion(_stopping_criterion)
          , verbosity_level(verbosity), max_iter(iter){

        }

        tools::shared_ptr<umintl::direction<BackendType> > direction;
        tools::shared_ptr<umintl::direction<BackendType> > fallback_direction;
        tools::shared_ptr<umintl::line_search<BackendType> > line_search;
        tools::shared_ptr<umintl::stopping_criterion<BackendType> > stopping_criterion;

        evaluation_policies_type evaluation_policies;

        double tolerance;

        unsigned int verbosity_level;
        unsigned int max_iter;

    private:

        std::string info() const{
          std::ostringstream oss;
          oss << "Verbosity Level : " << verbosity_level << std::endl;
          oss << "Maximum number of iterations : " << max_iter << std::endl;
          oss << "Direction : " << typeid(*direction).name() << std::endl;
          oss << "Line Search : " << typeid(*line_search).name() << std::endl;
          return oss.str();
        }

        optimization_result terminate(optimization_result::termination_cause_type termination_cause, typename BackendType::VectorType & res, std::size_t N, optimization_context<BackendType> & context){
            optimization_result result;
            BackendType::copy(N,context.x(),res);
            result.f = context.val();
            result.iteration = context.iter();
            result.n_functions_eval = context.fun().n_value_computations();
            result.n_gradient_eval = context.fun().n_gradient_computations();
            result.termination_cause = termination_cause;

            clean_all(context);

            return result;
        }

        void init_all(optimization_context<BackendType> & c){
            direction->init(c);
            line_search->init(c);
            stopping_criterion->init(c);
            fallback_direction->init(c);
        }

        void clean_all(optimization_context<BackendType> & c){
            direction->clean(c);
            line_search->clean(c);
            stopping_criterion->clean(c);
            fallback_direction->clean(c);
        }

    public:
        template<class Fun>
        optimization_result operator()(typename BackendType::VectorType & res, Fun & fun, typename BackendType::VectorType const & x0, std::size_t N){
            typedef typename BackendType::VectorType VectorType;

            tools::shared_ptr<umintl::direction<BackendType> > current_direction = direction;
            line_search_result<BackendType> search_res(N);
            optimization_context<BackendType> c(x0, N, new detail::function_wrapper_impl<BackendType, Fun>(fun,N,evaluation_policies));

            init_all(c);

            if(verbosity_level >= 1)
                std::cout << info() << std::endl;

            //First evaluation
            c.fun().compute_value_gradient(0, c.x(), c.val(), c.g());

            //Main loop
            for( ; c.iter() < max_iter ; ++c.iter()){

                if(verbosity_level >= 2 ){
                    std::cout << "Ieration  " << c.iter() << " | "
                              << "cost : " << c.val()
                              << "| NVal : " << c.fun().n_value_computations()
                              << "| NGrad : " << c.fun().n_gradient_computations();
                    if(unsigned int NHv = c.fun().n_hessian_vector_product_computations())
                     std::cout<< "| NHv : " << NHv ;
                    std::cout << std::endl;
                }

                current_direction = direction;
                if(c.is_reinitializing()){
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

                if(search_res.has_failed)
                    return terminate(optimization_result::LINE_SEARCH_FAILED, res, N, c);

                c.alpha() = search_res.best_alpha;

                BackendType::copy(N,c.x(),c.xm1());
                BackendType::copy(N,search_res.best_x,c.x());

                BackendType::copy(N,c.g(),c.gm1());
                BackendType::copy(N,search_res.best_g,c.g());

                c.valm1() = c.val();
                c.val() = search_res.best_phi;

                if((*stopping_criterion)(c))
                    return terminate(optimization_result::STOPPING_CRITERION, res, N, c);
            }

            return terminate(optimization_result::MAX_ITERATION_REACHED, res, N, c);
        }
    };


}

#endif
