/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_HPP_

#include "fmincl/utils.hpp"

namespace fmincl{

struct cg_restart{
    template<class BackendType>
    struct implementation : implementation_base<BackendType>{
        implementation(detail::optimization_context<BackendType> & context) : implementation_base<BackendType>(context){ }
        virtual bool operator()(void) = 0;
    };
    virtual ~cg_restart(){ }
};

struct no_restart : public cg_restart{
    template<class BackendType>
    struct implementation : public cg_restart::implementation<BackendType>{
        implementation(no_restart const & /*tag*/, detail::optimization_context<BackendType> & context) : cg_restart::implementation<BackendType>(context){ }
        bool operator()() { return false; }
    };
};


}

#endif
