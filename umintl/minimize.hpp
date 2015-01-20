/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_MINIMIZE_HPP_
#define UMINTL_MINIMIZE_HPP_

#include "umintl/optimization_result.hpp"

#include "umintl/model_base.hpp"

#include "umintl/function_wrapper.hpp"
#include "umintl/optimization_context.hpp"

#include "umintl/directions/conjugate_gradient.hpp"
#include "umintl/directions/quasi_newton.hpp"
#include "umintl/directions/low_memory_quasi_newton.hpp"
#include "umintl/directions/steepest_descent.hpp"
#include "umintl/directions/truncated_newton.hpp"

#include "umintl/line_search/strong_wolfe_powell.hpp"

#include "umintl/stopping_criterion/value_treshold.hpp"
#include "umintl/stopping_criterion/gradient_treshold.hpp"

#include <iomanip>
#include <sstream>

namespace umintl{

    /** @brief The minimizer class
     *
     */

    class minimizer
    {
    public:

        /** @brief The constructor
         *
         * @param _direction the descent direction used by the minimizer
         * @param _stopping_criterion the stopping criterion
         * @param _max_iter the maximum number of iterations
         * @param _verbosity_level the verbosity level
         */
        minimizer(umintl::direction * _direction = new quasi_newton()
                             , umintl::stopping_criterion * _stopping_criterion = new gradient_treshold()
                             , unsigned int _max_iter = 1024, unsigned int _verbosity_level = 0) :
            direction(_direction)
          , line_search(new strong_wolfe_powell())
          , stopping_criterion(_stopping_criterion)
          , model(new deterministic())
          , hessian_vector_product_computation(CENTERED_DIFFERENCE)
          , verbosity_level(_verbosity_level), max_iter(_max_iter){

        }

        tools::shared_ptr<umintl::direction > direction;
        tools::shared_ptr<umintl::line_search > line_search;
        tools::shared_ptr<umintl::stopping_criterion > stopping_criterion;
        tools::shared_ptr< model_base > model;
        computation_type hessian_vector_product_computation;

        double tolerance;

        unsigned int verbosity_level;
        unsigned int max_iter;

    private:

        /** @brief Get a brief info string on the minimizer
         *
         *  @return String containing the verbosity level, maximum number of iteration, and the direction used
         */
        std::string info() const{
          std::ostringstream oss;
          oss << "Verbosity Level : " << verbosity_level << std::endl;
          oss << "Maximum number of iterations : " << max_iter << std::endl;
          oss << "Direction : " << direction->info() << std::endl;
          return oss.str();
        }

        /** @brief Clean memory and terminate the optimization result
         *
         *  @return Optimization result
         */
        optimization_result terminate(optimization_result::termination_cause_type termination_cause, atidlas::array & res, optimization_context & context){
            optimization_result result;
            res = context.x();
            result.f = context.val();
            result.iteration = context.iter();
            result.n_functions_eval = context.fun().n_value_computations();
            result.n_gradient_eval = context.fun().n_gradient_computations();
            result.termination_cause = termination_cause;
            clean_all(context);
            return result;
        }

        /** @brief Init the components of the procedure (ie allocate memory for the temporaries, typically)
         */
        void init_all(optimization_context & c){
            direction->init(c);
            line_search->init(c);
            stopping_criterion->init(c);
        }

        /** @brief Clean the components of the procedure (ie free memory for the temporaries, typically)
         */
        void clean_all(optimization_context & c){
            direction->clean(c);
            line_search->clean(c);
            stopping_criterion->clean(c);
        }

    public:
        template<class Fun>
        optimization_result operator()(atidlas::array & res, Fun & fun, atidlas::array const & x0, std::size_t N){

            tools::shared_ptr<umintl::direction > steepest_descent(new umintl::steepest_descent());
            line_search_result search_res(N);

            optimization_context c(x0, *model, new detail::function_wrapper_impl<Fun>(fun,N,hessian_vector_product_computation));

            init_all(c);

            if(verbosity_level >= 1)
                std::cout << info() << std::endl;


            tools::shared_ptr<umintl::direction > current_direction;
            if(dynamic_cast<truncated_newton * >(direction.get()))
              current_direction = steepest_descent;
            else
              current_direction = steepest_descent;

            //Main loop
            c.fun().compute_value_gradient(c.x(), c.val(), c.g(), c.model().get_value_gradient_tag());
            for( ; c.iter() < max_iter ; ++c.iter()){
                if(verbosity_level >= 2 ){
                    std::cout << "Ieration  " << c.iter()
                              << "| cost : " << c.val()
                              << "| NVal : " << c.fun().n_value_computations()
                              << "| NGrad : " << c.fun().n_gradient_computations();
                    if(unsigned int NHv = c.fun().n_hessian_vector_product_computations())
                     std::cout<< "| NHv : " << NHv ;
                    if(unsigned int ND = c.fun().n_datapoints_accessed())
                     std::cout << "| NAccesses " << (float)ND;
                    std::cout << std::endl;
                }

                (*current_direction)(c);

                c.dphi_0() = atidlas::value_scalar(atidlas::dot(c.p(), c.g()));
                //Not a descent direction...
                if(c.dphi_0()>0){
                    //current_direction->reset(c);
                    current_direction = steepest_descent;
                    (*current_direction)(c);
                    c.dphi_0() = atidlas::value_scalar(atidlas::dot(c.p(), c.g()));
                }

                (*line_search)(search_res, current_direction.get(), c);

                if(search_res.has_failed){
                    return terminate(optimization_result::LINE_SEARCH_FAILED, res, c);
                }

                c.alpha() = search_res.best_alpha;
                c.xm1() = c.x();
                c.x() = search_res.best_x;
                c.gm1() = c.g();
                c.g() = search_res.best_g;
                c.valm1() = c.val();
                c.val() = search_res.best_phi;

                if((*stopping_criterion)(c)){
                    return terminate(optimization_result::STOPPING_CRITERION, res, c);
                }
                current_direction = direction;

                if(model->update(c))
                  c.fun().compute_value_gradient(c.x(), c.val(), c.g(), c.model().get_value_gradient_tag());
            }

            return terminate(optimization_result::MAX_ITERATION_REACHED, res, c);
        }
    };


}

#endif
