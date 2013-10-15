/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/

#ifndef UMINTL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_
#define UMINTL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_

#include "forwards.h"
#include <cmath>

namespace umintl{

template<class BackendType>
struct fletcher_reeves : public cg_update<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;
    typedef typename BackendType::VectorType VectorType;
    ScalarType operator()(optimization_context<BackendType> & c){
        return BackendType::dot(c.N(),c.g(),c.g())/BackendType::dot(c.N(),c.gm1(),c.gm1());
    }
};


}

#endif
