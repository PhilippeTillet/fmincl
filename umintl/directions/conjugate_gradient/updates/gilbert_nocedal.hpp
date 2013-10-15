/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_CONJUGATE_GRADIENT_UPDATES_GILBERT_NOCEDAL_HPP_
#define UMINTL_CONJUGATE_GRADIENT_UPDATES_GILBERT_NOCEDAL_HPP_

#include "forwards.h"
#include "polak_ribiere.hpp"
#include "fletcher_reeves.hpp"
#include <cmath>

namespace umintl{

template<class BackendType>
struct gilbert_nocedal : public cg_update<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;

    void init(optimization_context<BackendType> & c){
        pr_.init(c);
        fr_.init(c);
    }

    void clean(optimization_context<BackendType> & c){
        pr_.clean(c);
        fr_.clean(c);
    }

    ScalarType operator()(optimization_context<BackendType> & context){
        ScalarType betaPR = pr_(context);
        ScalarType betaFR = fr_(context);
        return std::min(betaPR,betaFR);
    }
private:
    polak_ribiere<BackendType> pr_;
    fletcher_reeves<BackendType> fr_;
};


}

#endif
