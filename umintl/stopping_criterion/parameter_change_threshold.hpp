/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_STOPPING_CRITERION_PARAMETER_CHANGE_TRESHOLD_HPP_
#define UMINTL_STOPPING_CRITERION_PARAMETER_CHANGE_TRESHOLD_HPP_

#include <cmath>

#include "umintl/optimization_context.hpp"
#include "forwards.h"

namespace umintl{

/** @brief parameter-based stopping criterion
 *
 *  Stops the optimization procedure when the euclidian norm of the change of parameters accross two successive iterations is below a threshold
 */

struct parameter_change_threshold : public stopping_criterion{
    parameter_change_threshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;
    bool operator()(optimization_context & c)
    { return  atidlas::norm(c.x() - c.xm1()) < tolerance; }
};

}

#endif
