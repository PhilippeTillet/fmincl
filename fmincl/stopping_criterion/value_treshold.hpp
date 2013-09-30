/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_
#define FMINCL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_

#include <cmath>

#include "fmincl/utils.hpp"
#include "forwards.h"

namespace fmincl{

template<class BackendType>
struct value_treshold : public stopping_criterion<BackendType>{
    value_treshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;

    bool operator()(detail::optimization_context<BackendType> & c){
        return std::fabs(c.val() - c.valm1()) < tolerance;
    }
};

}

#endif
