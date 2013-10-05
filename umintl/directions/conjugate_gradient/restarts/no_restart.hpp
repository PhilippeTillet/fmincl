/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_NO_RESTART_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_NO_RESTART_HPP_

#include "forwards.h"

namespace umintl{

template<class BackendType>
struct no_restart : public cg_restart<BackendType>{
    bool operator()(detail::optimization_context<BackendType> &) { return false; }
};


}

#endif
