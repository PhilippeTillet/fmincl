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
#include "tools/shared_ptr.hpp"
#include "fmincl/directions.hpp"
#include "fmincl/line_search.hpp"

namespace fmincl{

  struct optimization_options{
      optimization_options(unsigned int verbosity = 0, unsigned int iter = 512) : verbosity_level(verbosity), max_iter(iter){ }
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
      mutable tools::shared_ptr<line_search_tag> line_search;
      unsigned int verbosity_level;
      unsigned int max_iter;
  };

}

#endif
