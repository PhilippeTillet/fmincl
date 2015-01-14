/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_DIRECTIONS_FORWARDS_H
#define UMINTL_DIRECTIONS_FORWARDS_H

#include "umintl/optimization_context.hpp"

namespace umintl{


struct direction{
    
    virtual ~direction(){ }
    virtual void operator()(optimization_context &) = 0;
    virtual std::string info() const = 0;
    virtual void init(optimization_context &){ }
    virtual void clean(optimization_context &){ }
};


}

#endif
