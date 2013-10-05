/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_OPTIMIZATION_RESULT_HPP_
#define UMINTL_OPTIMIZATION_RESULT_HPP_

#include <cstddef>

namespace umintl{

  struct optimization_result{
  private:
  public:
      enum termination_cause_type{
          LINE_SEARCH_FAILED,
          STOPPING_CRITERION,
          MAX_ITERATION_REACHED
      };

      double f;
      std::size_t iteration;
      std::size_t n_functions_eval;
      std::size_t n_gradient_eval;
      termination_cause_type termination_cause;
  };

}

#endif
