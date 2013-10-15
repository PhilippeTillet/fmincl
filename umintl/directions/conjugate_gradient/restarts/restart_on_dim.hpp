/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_RESTART_ON_DIM_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_RESTART_ON_DIM_HPP_

#include "forwards.h"

namespace umintl{

template<class BackendType>
struct restart_on_dim : public cg_restart<BackendType>{
    bool operator()(optimization_context<BackendType> & c) { return c.iter()==c.N(); }
};


}

#endif
