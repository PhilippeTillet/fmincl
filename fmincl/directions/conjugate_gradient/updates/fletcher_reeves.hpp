/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_

#include "forwards.h"
#include <cmath>

namespace fmincl{

template<class BackendType>
struct fletcher_reeves : public cg_update<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    ScalarType operator()(detail::optimization_context<BackendType> & c){
        return BackendType::dot(c.N(),c.g(),c.g())/BackendType::dot(c.N(),c.gm1(),c.gm1());
    }
};


}

#endif
