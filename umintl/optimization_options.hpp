/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_OPTIMIZATION_OPTIONS_HPP_
#define UMINTL_OPTIMIZATION_OPTIONS_HPP_

#include <typeinfo>
#include <sstream>

#include "umintl/tools/shared_ptr.hpp"

#include "umintl/directions/forwards.h"
#include "umintl/directions/quasi_newton.hpp"
#include "umintl/directions/conjugate_gradient.hpp"

#include "umintl/line_search/strong_wolfe_powell.hpp"

#include "umintl/stopping_criterion/forwards.h"
#include "umintl/stopping_criterion/gradient_treshold.hpp"

namespace umintl{

  template<class BackendType>
  struct optimization_options{
      optimization_options(umintl::direction<BackendType> * _direction = new quasi_newton<BackendType>()
                           , umintl::stopping_criterion<BackendType> * _stopping_criterion = new gradient_treshold<BackendType>()
                           , unsigned int iter = 1024, unsigned int verbosity = 0) : direction(_direction), line_search(new strong_wolfe_powell<BackendType>()), stopping_criterion(_stopping_criterion), verbosity_level(verbosity), max_iter(iter){

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
      mutable tools::shared_ptr<umintl::direction<BackendType> > direction;
      mutable tools::shared_ptr<umintl::line_search<BackendType> > line_search;
      mutable tools::shared_ptr<umintl::stopping_criterion<BackendType> > stopping_criterion;

      double tolerance;

      unsigned int verbosity_level;
      unsigned int max_iter;
  };

}

#endif
