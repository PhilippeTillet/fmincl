/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_OPTIMIZATION_OPTIONS_HPP_
#define FMINCL_OPTIMIZATION_OPTIONS_HPP_

#include <typeinfo>
#include <sstream>

#include "fmincl/tools/shared_ptr.hpp"

#include "fmincl/directions/forwards.h"
#include "fmincl/directions/quasi_newton.hpp"
#include "fmincl/directions/conjugate_gradient.hpp"

#include "fmincl/line_search/strong_wolfe_powell.hpp"

#include "fmincl/stopping_criterion/forwards.h"
#include "fmincl/stopping_criterion/gradient_treshold.hpp"

namespace fmincl{

  struct minibatch_options{
      minibatch_options(std::size_t _n_minibatches, std::size_t _iteration_per_minibatch) : n_minibatches(_n_minibatches), iteration_per_minibatch(_iteration_per_minibatch){ }
      std::size_t n_minibatches;
      std::size_t iteration_per_minibatch;
  };

  struct optimization_options{
      optimization_options(fmincl::direction * _direction = new quasi_newton(), fmincl::stopping_criterion * _stopping_criterion = new gradient_treshold(), unsigned int iter = 1024, unsigned int verbosity = 0) : direction(_direction), line_search(new strong_wolfe_powell()), stopping_criterion(_stopping_criterion), verbosity_level(verbosity), max_iter(iter){

      }
      std::string info() const{
        std::ostringstream oss;
        oss << "Verbosity Level : " << verbosity_level << std::endl;
        oss << "Maximum number of iterations : " << max_iter << std::endl;
        oss << "Direction : " << typeid(*direction).name() << std::endl;
        oss << "Line Search : " << typeid(*line_search).name() << std::endl;
        oss << std::endl;
        return oss.str();
      }
      mutable tools::shared_ptr<fmincl::direction> direction;
      mutable tools::shared_ptr<fmincl::line_search> line_search;
      mutable tools::shared_ptr<fmincl::stopping_criterion> stopping_criterion;

      double tolerance;

      unsigned int verbosity_level;
      unsigned int max_iter;
  };

}

#endif