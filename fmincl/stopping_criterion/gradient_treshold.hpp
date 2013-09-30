/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_
#define FMINCL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_

#include <cmath>

#include "fmincl/utils.hpp"

#include "forwards.h"

namespace fmincl{

template<class BackendType>
struct gradient_treshold : public stopping_criterion<BackendType>{
    gradient_treshold(double _tolerance = 1e-4) : tolerance(_tolerance){ }
    double tolerance;

    bool operator()(detail::optimization_context<BackendType> & c){
        return BackendType::nrm2(c.N(),c.g()) < tolerance;
    }
};



}

#endif
