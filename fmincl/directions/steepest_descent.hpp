/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_STEEPEST_DESCENT_HPP_
#define FMINCL_DIRECTIONS_STEEPEST_DESCENT_HPP_

#include "fmincl/utils.hpp"

#include "fmincl/mapping.hpp"

#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/directions/forwards.h"


namespace fmincl{

template<class BackendType>
struct steepest_descent : public direction<BackendType>{
    void operator()(detail::optimization_context<BackendType> & c){
        std::size_t N = c.N();
        BackendType::copy(N,c.g(),c.p());
        BackendType::scale(N,-1,c.p());
    }
};

}

#endif
