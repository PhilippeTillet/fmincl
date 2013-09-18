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
    struct implementation {
        virtual bool operator()(detail::optimization_context<BackendType> & c) = 0;
        virtual ~implementation(){ }
    };
    virtual ~cg_restart(){ }
};

struct no_restart : public cg_restart{
    template<class BackendType>
    struct implementation : public cg_restart::implementation<BackendType>{
        implementation(no_restart const & /*tag*/, detail::optimization_context<BackendType> & context){ }
        bool operator()(detail::optimization_context<BackendType> & c) { return false; }
    };
};

struct restart_on_dim : public cg_restart{
    template<class BackendType>
    struct implementation : public cg_restart::implementation<BackendType>{
        implementation(restart_on_dim const & /*tag*/, detail::optimization_context<BackendType> & context){ }
        bool operator()(detail::optimization_context<BackendType> & c) { return c.iter()==c.dim(); }
    };
};

struct restart_not_sufficient_descent : public cg_restart{
    template<class BackendType>
    struct implementation : public cg_restart::implementation<BackendType>{
        implementation(restart_on_dim const & /*tag*/, detail::optimization_context<BackendType> & context){ }
        bool operator()(detail::optimization_context<BackendType> & c) { return c.iter()==c.dim(); }
    };
};

}

#endif
