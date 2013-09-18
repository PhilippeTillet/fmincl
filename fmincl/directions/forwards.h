/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_FORWARDS_H
#define FMINCL_DIRECTIONS_FORWARDS_H

#include "fmincl/utils.hpp"

namespace fmincl{

struct direction{
    template<class BackendType>
    struct implementation{
        virtual void operator()(detail::optimization_context<BackendType> &) = 0;
        virtual bool restart(detail::optimization_context<BackendType> &){ return false; }
        virtual ~implementation(){ }
    };


    virtual ~direction(){}
};


}

#endif
