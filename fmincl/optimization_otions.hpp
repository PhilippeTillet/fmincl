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
#include "fmincl/directions.hpp"
#include "fmincl/line_search.hpp"
#include "fmincl/stopping_criterion.hpp"

namespace fmincl{

  struct optimization_options{
      optimization_options(direction_tag * _direction = new quasi_newton_tag(), stopping_criterion_tag * _stopping_criterion = new gradient_based_stopping_tag(), unsigned int iter = 1024, unsigned int verbosity = 0) : direction(_direction), stopping_criterion(_stopping_criterion), verbosity_level(verbosity), max_iter(iter){ }
      std::string info() const{
        std::ostringstream oss;
        oss << "Verbosity Level : " << verbosity_level << std::endl;
        oss << "Max Iter : " << max_iter << std::endl;
        oss << "Direction : " << typeid(*direction).name() << std::endl;
        oss << "Line Search : " << typeid(*line_search).name() << std::endl;
        oss << std::endl;
        return oss.str();
      }
      mutable tools::shared_ptr<direction_tag> direction;
      mutable tools::shared_ptr<stopping_criterion_tag> stopping_criterion;
      mutable tools::shared_ptr<line_search_tag> line_search;
      double tolerance;
      unsigned int verbosity_level;
      unsigned int max_iter;
  };

}

#endif
