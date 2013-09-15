/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_
#define FMINCL_STOPPING_CRITERION_VALUE_TRESHOLD_HPP_

#include <cmath>

#include "fmincl/utils.hpp"
#include "forwards.h"

namespace fmincl{

struct value_treshold : public stopping_criterion{
    value_treshold(double _tolerance = 1e-5) : tolerance(_tolerance){ }
    double tolerance;

    template<class BackendType>
    struct implementation : public stopping_criterion::implementation<BackendType>{
        implementation(value_treshold const & _tag, detail::optimization_context<BackendType> & context) : context_(context), tag(_tag){ }
        bool operator()(){
            return std::fabs(context_.val() - context_.valm1()) < tag.tolerance;
        }
    private:
        detail::optimization_context<BackendType> & context_;
        value_treshold const & tag;
    };
};

}

#endif
