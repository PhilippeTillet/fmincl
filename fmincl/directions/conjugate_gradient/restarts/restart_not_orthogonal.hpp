/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_NOT_ORTHOGONAL_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_RESTARTS_NOT_ORTHOGONAL_HPP_

#include "forwards.h"

namespace fmincl{

struct restart_not_orthogonal : public cg_restart{
    restart_not_orthogonal(double _threshold = 0.1) : threshold(_threshold){ }
    double threshold;

    template<class BackendType>
    struct implementation : public cg_restart::implementation<BackendType>{
        implementation(restart_not_orthogonal const & tag, detail::optimization_context<BackendType> &) : tag_(tag){ }
        bool operator()(detail::optimization_context<BackendType> & c)
        {
            return std::abs(BackendType::dot(c.N(),c.g(),c.gm1()))/BackendType::dot(c.N(),c.g(),c.g()) > tag_.threshold;
        }
    private:
        restart_not_orthogonal const & tag_;
    };
};

}

#endif
