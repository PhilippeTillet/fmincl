/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_
#define FMINCL_STOPPING_CRITERION_GRADIENT_TRESHOLD_HPP_

#include <cmath>

#include "fmincl/utils.hpp"

#include "forwards.h"

namespace fmincl{

struct gradient_treshold : public stopping_criterion{
    gradient_treshold(double _tolerance = 1e-4) : tolerance(_tolerance){ }
    double tolerance;

    template<class BackendType>
    struct implementation : public stopping_criterion::implementation<BackendType>{
        implementation(gradient_treshold const & _tag, detail::optimization_context<BackendType> &) : tag(_tag){ }
        bool operator()(detail::optimization_context<BackendType> & c){
            return BackendType::nrm2(c.N(),c.g()) < tag.tolerance;
        }
    private:
        gradient_treshold const & tag;
    };

};



}

#endif
