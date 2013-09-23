/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_
#define FMINCL_CONJUGATE_GRADIENT_UPDATES_FLETCHER_REEVES_HPP_

#include "forwards.h"
#include <cmath>

namespace fmincl{

struct fletcher_reeves : public cg_update{
    template<class BackendType>
    struct implementation : public cg_update::implementation<BackendType>{
    private:
        typedef typename BackendType::ScalarType ScalarType;
        typedef typename BackendType::VectorType VectorType;
    public:
        implementation(fletcher_reeves const &, detail::optimization_context<BackendType> & context){
            N_ = context.N();
        }
        ScalarType operator()(detail::optimization_context<BackendType> & c){
            return BackendType::dot(N_,c.g(),c.g())/BackendType::dot(N_,c.gm1(),c.gm1());
        }
    private:
        std::size_t N_;
    };
};


}

#endif
