/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_
#define UMINTL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_

#include "umintl/optimization_context.hpp"

#include <cmath>

namespace umintl{

template<class BackendType>
struct cg_update{
    virtual ~cg_update(){ }
    virtual void init(optimization_context<BackendType> &){ }
    virtual void clean(optimization_context<BackendType> &){ }
    virtual typename BackendType::ScalarType operator()(optimization_context<BackendType> &) = 0;
};

}

#endif
