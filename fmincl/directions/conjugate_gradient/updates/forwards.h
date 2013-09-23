/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_

#include "fmincl/utils.hpp"

#include <cmath>

namespace fmincl{

struct cg_update{
    virtual ~cg_update(){ }

    template<class BackendType>
    struct implementation{
        typedef typename BackendType::ScalarType ScalarType;
        virtual ScalarType operator()(detail::optimization_context<BackendType> &) = 0;
        virtual ~implementation(){ }
    };
};

}

#endif
