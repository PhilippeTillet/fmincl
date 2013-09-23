/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_GILBERT_NOCEDAL_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_GILBERT_NOCEDAL_HPP_

#include "forwards.h"
#include <cmath>

namespace fmincl{

struct gilbert_nocedal : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(gilbert_nocedal const &, detail::optimization_context<BackendType> &){ }
        ScalarType operator()(detail::optimization_context<BackendType> & context){
            ScalarType betaPR = polak_ribiere::implementation<BackendType>(polak_ribiere(),context)(context);
            ScalarType betaFR = fletcher_reeves::implementation<BackendType>(fletcher_reeves(),context)(context);
            return std::min(betaPR,betaFR);
        }
    };
};


}

#endif
