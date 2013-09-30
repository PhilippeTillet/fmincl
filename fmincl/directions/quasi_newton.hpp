/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>


#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/utils.hpp"

#include "forwards.h"
#include "quasi_newton/updates/lbfgs.hpp"
#include "quasi_newton/updates/bfgs.hpp"



namespace fmincl{

template<class BackendType>
struct quasi_newton : public direction<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;

    quasi_newton(qn_update<BackendType> * _update = new lbfgs<BackendType>()) : update(_update){ }

    virtual void init(detail::optimization_context<BackendType> & c){
        update->init(c);
    }
    virtual void clean(detail::optimization_context<BackendType> & c){
        update->clean(c);
    }

    virtual ScalarType line_search_first_trial(detail::optimization_context<BackendType> &){
        return 1;
    }

    virtual void operator()(detail::optimization_context<BackendType> & context){
        (*update)(context);
    }

    tools::shared_ptr<qn_update<BackendType> > update;
};

}

#endif
