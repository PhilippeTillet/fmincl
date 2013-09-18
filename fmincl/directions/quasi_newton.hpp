/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_
#define FMINCL_DIRECTIONS_QUASI_NEWTON_HPP_

#include <vector>
#include <cmath>


#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/utils.hpp"

#include "forwards.h"
#include "quasi_newton/updates/lbfgs.hpp"
#include "quasi_newton/updates/bfgs.hpp"



namespace fmincl{

struct quasi_newton : public direction{
    quasi_newton(qn_update * _update = new lbfgs()) : update(_update){ }
    tools::shared_ptr<qn_update> update;

    template<class BackendType>
    class implementation : public direction::implementation<BackendType>{
        typedef implementation_of<BackendType,qn_update,bfgs,lbfgs> update_mapping;
      public:
        implementation(quasi_newton const & tag, detail::optimization_context<BackendType> & context) : update(update_mapping::create(*tag.update, context)){ }
        virtual void operator()(detail::optimization_context<BackendType> & context)
        {
            (*update)(context);
        }
    private:
        tools::shared_ptr<qn_update::implementation<BackendType> > update;
    };
};

}

#endif
