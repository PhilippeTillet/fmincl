/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_RESTART_ON_DIM_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_RESTART_ON_DIM_HPP_

#include "forwards.h"

namespace fmincl{

template<class BackendType>
struct restart_on_dim : public cg_restart<BackendType>{
    bool operator()(detail::optimization_context<BackendType> & c) { return c.iter()==c.N(); }
};


}

#endif
