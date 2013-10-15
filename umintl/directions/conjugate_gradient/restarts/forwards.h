/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_FORWARDS_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_FORWARDS_HPP_

#include "umintl/utils.hpp"

namespace umintl{

template<class BackendType>
struct cg_restart{
    virtual ~cg_restart(){ }
    virtual void init(detail::optimization_context<BackendType> &){ }
    virtual void clean(detail::optimization_context<BackendType> &){ }
    virtual bool operator()(detail::optimization_context<BackendType> & c) = 0;
};

}

#endif
