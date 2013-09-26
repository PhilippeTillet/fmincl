/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_FORWARDS_H
#define FMINCL_DIRECTIONS_QUASI_NEWTON_FORWARDS_H

#include "fmincl/mapping.hpp"
#include "fmincl/utils.hpp"

namespace fmincl{

struct qn_update{
    template<class BackendType>
    struct implementation{
        virtual void operator()(detail::optimization_context<BackendType> &) = 0;
        virtual ~implementation(){ }
    };

    virtual ~qn_update(){ }
};

}

#endif
