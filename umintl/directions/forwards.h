/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_FORWARDS_H
#define UMINTL_DIRECTIONS_FORWARDS_H

#include "umintl/optimization_context.hpp"

namespace umintl{

template<class BackendType>
struct direction
{
    typedef typename BackendType::ScalarType ScalarType;
    virtual ~direction(){ }
    virtual void operator()(optimization_context<BackendType> &) = 0;
    virtual std::string info() const = 0;
    virtual void init(optimization_context<BackendType> &){ }
    virtual void clean(optimization_context<BackendType> &){ }
    virtual ScalarType step_size(optimization_context<BackendType> &) = 0;
    virtual void strong_wolfe_powell_parameters(ScalarType & c1, ScalarType & c2) const = 0;
};


}

#endif
