/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_STOPPING_CRITERION_FORWARDS_H
#define UMINTL_STOPPING_CRITERION_FORWARDS_H


#include "umintl/optimization_context.hpp"

namespace umintl{

/** @brief Base class for a stopping criterion */

struct stopping_criterion{
    virtual ~stopping_criterion(){ }
    virtual void init(optimization_context &){ }
    virtual void clean(optimization_context &){ }
    virtual bool operator()(optimization_context & context) = 0;
};



}

#endif
