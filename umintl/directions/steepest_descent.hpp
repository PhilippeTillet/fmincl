/* ===========================
  Copyright (c) 2013 Philippe Tillet
  UMinTL - Unconstrained Minimization Template Library

  License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef UMINTL_DIRECTIONS_STEEPEST_DESCENT_HPP_
#define UMINTL_DIRECTIONS_STEEPEST_DESCENT_HPP_

#include "umintl/optimization_context.hpp"

#include "umintl/tools/shared_ptr.hpp"
#include "umintl/directions/forwards.h"


namespace umintl{

template<class BackendType>
struct steepest_descent : public direction<BackendType>{
    typedef typename BackendType::ScalarType ScalarType;

    virtual std::string info() const{
        return "Steepest Descent";
    }

    void operator()(optimization_context<BackendType> & c){
        std::size_t N = c.N();
        BackendType::copy(N,c.g(),c.p());
        BackendType::scale(N,-1,c.p());
    }

    ScalarType step_size(optimization_context<BackendType> & c) {
        return std::min((typename BackendType::ScalarType)(1.0),1/BackendType::asum(c.N(),c.g()));
    }

    void strong_wolfe_powell_parameters(ScalarType & c1, ScalarType & c2) const{
        c1 = 1e-4;
        c2 = 0.2;
    }

};

}

#endif
