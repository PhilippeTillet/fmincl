/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_FORWARDS_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_FORWARDS_HPP_

#include "fmincl/utils.hpp"

namespace fmincl{

struct cg_restart{
    template<class BackendType>
    struct implementation {
        virtual bool operator()(detail::optimization_context<BackendType> & c) = 0;
        virtual ~implementation(){ }
    };
    virtual ~cg_restart(){ }
};

}

#endif
