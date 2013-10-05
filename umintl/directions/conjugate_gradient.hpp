/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * umintl - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_
#define UMINTL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_

#include "umintl/utils.hpp"

#include "umintl/mapping.hpp"

#include "umintl/tools/shared_ptr.hpp"
#include "umintl/directions/forwards.h"

#include "conjugate_gradient/restarts/no_restart.hpp"
#include "conjugate_gradient/restarts/restart_not_orthogonal.hpp"
#include "conjugate_gradient/restarts/restart_on_dim.hpp"

#include "conjugate_gradient/updates/polak_ribiere.hpp"
#include "conjugate_gradient/updates/fletcher_reeves.hpp"
#include "conjugate_gradient/updates/gilbert_nocedal.hpp"


namespace umintl{

template<class BackendType>
struct conjugate_gradient : public direction<BackendType>{
    typedef typename BackendType::VectorType VectorType;
    typedef typename BackendType::ScalarType ScalarType;
    conjugate_gradient(cg_update<BackendType> * _update = new gilbert_nocedal<BackendType>(), cg_restart<BackendType> * _restart = new restart_not_orthogonal<BackendType>()) : update(_update), restart(_restart){ }

    virtual void init(detail::optimization_context<BackendType> & c){
        update->init(c);
        restart->init(c);
    }

    virtual void clean(detail::optimization_context<BackendType> & c){
        update->clean(c);
        restart->clean(c);
    }

    void operator()(detail::optimization_context<BackendType> & c){
        ScalarType beta;
        if((*restart)(c))
            beta = 0;
        else
            beta = (*update)(c);
        BackendType::scale(c.N(),beta,c.p());
        BackendType::axpy(c.N(),-1,c.g(),c.p());
    }

    tools::shared_ptr< cg_update<BackendType> > update;
    tools::shared_ptr< cg_restart<BackendType> > restart;
};

}

#endif
