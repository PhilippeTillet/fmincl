/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_
#define UMINTL_CONJUGATE_GRADIENT_UPDATES_FORWARDS_HPP_

#include "umintl/utils.hpp"

#include <cmath>

namespace umintl{

template<class BackendType>
struct cg_update{
    virtual ~cg_update(){ }
    virtual void init(detail::optimization_context<BackendType> &){ }
    virtual void clean(detail::optimization_context<BackendType> &){ }
    virtual typename BackendType::ScalarType operator()(detail::optimization_context<BackendType> &) = 0;
};

}

#endif
