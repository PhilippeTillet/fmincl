/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_
#define UMINTL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>


#include "umintl/tools/shared_ptr.hpp"
#include "umintl/optimization_context.hpp"

#include "forwards.h"
#include "quasi_newton/updates/lbfgs.hpp"
#include "quasi_newton/updates/bfgs.hpp"



namespace umintl{

template<class BackendType>
struct quasi_newton : public direction<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;

    quasi_newton(qn_update<BackendType> * _update = new lbfgs<BackendType>()) : update(_update){ }

    virtual void init(optimization_context<BackendType> & c){
        update->init(c);
    }
    virtual void clean(optimization_context<BackendType> & c){
        update->clean(c);
    }

    virtual ScalarType line_search_first_trial(optimization_context<BackendType> &){
        return 1;
    }

    virtual void operator()(optimization_context<BackendType> & context){
        (*update)(context);
    }

    tools::shared_ptr<qn_update<BackendType> > update;
};

}

#endif
