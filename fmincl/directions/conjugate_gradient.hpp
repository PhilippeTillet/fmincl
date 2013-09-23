/* ===========================
 *
 * Copyright (c) 2013 Philippe Tillet - National Chiao Tung University
 *
 * FMinCL - Unconstrained Function Minimization on OpenCL
 *
 * License : MIT X11 - See the LICENSE file in the root folder
 * ===========================*/


#ifndef FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_
#define FMINCL_DIRECTIONS_CONJUGATE_GRADIENT_HPP_

#include "fmincl/utils.hpp"

#include "fmincl/mapping.hpp"

#include "fmincl/tools/shared_ptr.hpp"
#include "fmincl/directions/forwards.h"

#include "conjugate_gradient/restarts/no_restart.hpp"
#include "conjugate_gradient/restarts/restart_not_orthogonal.hpp"
#include "conjugate_gradient/restarts/restart_on_dim.hpp"

#include "conjugate_gradient/updates/polak_ribiere.hpp"
#include "conjugate_gradient/updates/fletcher_reeves.hpp"
#include "conjugate_gradient/updates/gilbert_nocedal.hpp"


namespace fmincl{

struct conjugate_gradient : public direction{
    template<class BackendType>
    class implementation : public direction::implementation<BackendType>{
        typedef typename BackendType::VectorType VectorType;
        typedef typename BackendType::ScalarType ScalarType;
        typedef implementation_of<BackendType,cg_update,polak_ribiere,fletcher_reeves,gilbert_nocedal> update_mapping;
        typedef implementation_of<BackendType,cg_restart,no_restart,restart_not_orthogonal,restart_on_dim> restart_mapping;


    public:
        implementation(conjugate_gradient const & cg_params, detail::optimization_context<BackendType> & context) : update_implementation_(update_mapping::create(*cg_params.update, context))
                                                                                                                   ,restart_implementation_(restart_mapping::create(*cg_params.restart,context)){ }

        virtual bool restart(detail::optimization_context<BackendType> & c){
            return (*restart_implementation_)(c);
        }

        void operator()(detail::optimization_context<BackendType> & c){
          //p = -g + beta*p;
          ScalarType beta = (*update_implementation_)(c);
          BackendType::scale(c.N(),beta,c.p());
          BackendType::axpy(c.N(),-1,c.g(),c.p());
        }
    private:
        tools::shared_ptr<cg_update::implementation<BackendType> > update_implementation_;
        tools::shared_ptr<cg_restart::implementation<BackendType> > restart_implementation_;
    };


    conjugate_gradient(cg_update * _update = new gilbert_nocedal(), cg_restart * _restart = new restart_not_orthogonal()) : update(_update), restart(_restart){ }
    tools::shared_ptr<cg_update> update;
    tools::shared_ptr<cg_restart> restart;
};

}

#endif
