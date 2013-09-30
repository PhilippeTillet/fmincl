/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_FORWARDS_H
#define FMINCL_STOPPING_CRITERION_FORWARDS_H


#include "fmincl/utils.hpp"

namespace fmincl{

template<class BackendType>
struct stopping_criterion{
    virtual ~stopping_criterion(){ }
    virtual void init(detail::optimization_context<BackendType> &){ }
    virtual void clean(detail::optimization_context<BackendType> &){ }
    virtual bool operator()(detail::optimization_context<BackendType> & context) = 0;
};



}

#endif
